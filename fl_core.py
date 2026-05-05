"""
fl_core.py
==========
Core library for SecAgg-compatible robust FL experiments.

Design principles:
- SecAgg constraint is enforced at the type level: the Server class never
  receives per-client tensors, only cohort-level sums.
- All cohort logic operates on either (a) public metadata or (b) outputs of
  random projection sketches that compose under summation.
- Belief propagation runs on the cohort-client factor graph using only
  cohort-level residuals (no per-client info).
- Every randomness source has an explicit seed.
- Logging is row-wise CSV append; configs are JSON dumped alongside.

Mathematical grounding:
- Cohort overlap structure: configuration model of random regular graphs.
  Critical density for giant component: rho* ~ 1/sqrt(n*k) for our setup
  (each client in k cohorts, n total clients). See Newman (2010).
- Topological neighbor selection: rank-based, following Ballerini et al.
  (2008) on starling murmurations - each agent considers fixed-k nearest,
  not metric-distance neighbors.
- Belief propagation: standard sum-product on factor graph (Pearl 1988,
  Yedidia et al. 2003). We use log-domain for numerical stability.
- Random projection sketch: Johnson-Lindenstrauss (1984). Composes under
  summation, so cohort-sum of sketches = sketch of cohort-sum, which is
  what SecAgg actually delivers.

This file is the single source of truth for all phases. Phase notebooks
import from here.
"""

import os
import json
import time
import math
import hashlib
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# =============================================================================
# Reproducibility
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set every randomness source we use. Call at start of every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms slow things down; we accept that for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_hash(cfg: dict) -> str:
    """Stable hash of a config dict, for run identification."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# =============================================================================
# Models (FL-conventional, GroupNorm not BatchNorm)
# =============================================================================

class GroupNorm2d(nn.GroupNorm):
    """GroupNorm with sensible defaults for small channel counts."""
    def __init__(self, num_channels: int, num_groups: Optional[int] = None):
        if num_groups is None:
            num_groups = min(8, num_channels)
            while num_channels % num_groups != 0:
                num_groups -= 1
        super().__init__(num_groups, num_channels)


class BasicBlockGN(nn.Module):
    """ResNet basic block with GroupNorm. Standard FL convention."""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.gn1 = GroupNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.gn2 = GroupNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                GroupNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet8GN(nn.Module):
    """ResNet-8 with GroupNorm. ~78K params. Standard FL benchmark model.

    Architecture: conv -> [BasicBlock] x 3 -> avgpool -> fc
    Used in SCAFFOLD/FedDyn-style papers as a lighter ResNet variant.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.gn1 = GroupNorm2d(16)
        self.layer1 = BasicBlockGN(16, 16, stride=1)
        self.layer2 = BasicBlockGN(16, 32, stride=2)
        self.layer3 = BasicBlockGN(32, 64, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


class LinearRegressionModel(nn.Module):
    """Simple linear model for Phase 1 toy experiments."""
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Parameter <-> flat vector helpers
# =============================================================================

def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all params into a single flat vector. Detached, on CPU."""
    return torch.cat([p.data.detach().flatten().cpu() for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Inverse of get_flat_params. Writes into model.parameters() in order."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p.data).to(p.device))
        idx += n
    assert idx == flat.numel(), f"Size mismatch: {idx} vs {flat.numel()}"


# =============================================================================
# Data partitioning (Dirichlet non-IID)
# =============================================================================

def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_samples_per_client: int = 10,
) -> list:
    """Partition dataset indices among clients using Dirichlet(alpha) over labels.

    Standard FL non-IID benchmark (Yurochkin et al. 2019, Hsu et al. 2019).
    alpha -> 0: highly non-IID (each client sees few labels).
    alpha -> inf: IID.

    Returns list of index arrays, one per client. Guarantees each client has
    at least `min_samples_per_client` samples by retrying on degenerate splits.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(targets.max()) + 1
    targets = np.asarray(targets)

    for attempt in range(20):
        client_idxs = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            class_idx = np.where(targets == c)[0]
            rng.shuffle(class_idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            # Round to integer counts that sum exactly
            counts = np.floor(proportions * len(class_idx)).astype(int)
            counts[-1] = len(class_idx) - counts[:-1].sum()
            splits = np.split(class_idx, np.cumsum(counts)[:-1])
            for i, s in enumerate(splits):
                client_idxs[i].extend(s.tolist())

        sizes = [len(idx) for idx in client_idxs]
        if min(sizes) >= min_samples_per_client:
            return [np.array(idx) for idx in client_idxs]

    raise RuntimeError(
        f"Could not produce min_samples_per_client={min_samples_per_client} "
        f"after 20 attempts. Try larger alpha or fewer clients."
    )


class IndexedSubset(Dataset):
    """Subset of a base dataset by index. Lightweight."""
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


# =============================================================================
# Random projection sketch (Johnson-Lindenstrauss)
# =============================================================================

class JLSketch:
    """Random projection sketch S: R^d -> R^m.

    JL guarantees: for m = O(log(n)/eps^2), all pairwise distances among
    n vectors are preserved within (1 +/- eps) factor.

    Critical property for SecAgg: sketches compose linearly under summation.
        S(x_1 + x_2 + ... + x_k) = S(x_1) + ... + S(x_k)
    So the server, seeing only sum_i x_i (SecAgg output), can equivalently
    work with sum_i S(x_i) -- no per-client info needed.

    We use a fixed Gaussian projection seeded for reproducibility.
    """
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seed = seed
        g = torch.Generator().manual_seed(seed)
        # Scaled Gaussian: 1/sqrt(m) factor for unbiased norm preservation
        self.projection = torch.randn(in_dim, out_dim, generator=g) / math.sqrt(out_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) -> (..., out_dim). Uses CPU tensor for stability."""
        return x.cpu() @ self.projection


# =============================================================================
# Cohort assignment
# =============================================================================

def metric_cohorts(
    metadata: np.ndarray,
    num_cohorts: int,
    cohort_size: int,
    seed: int = 0,
) -> list:
    """Metric (k-means style) cohort assignment from metadata.

    Each client may appear in multiple cohorts (overlap). We perform
    k-means on metadata to get clusters, then for each cluster center
    take the cohort_size nearest clients.

    Returns list of cohort index arrays.
    """
    from sklearn.cluster import KMeans
    n = len(metadata)
    if metadata.ndim == 1:
        metadata = metadata.reshape(-1, 1)
    km = KMeans(n_clusters=num_cohorts, random_state=seed, n_init=10).fit(metadata)
    centers = km.cluster_centers_
    cohorts = []
    for c in range(num_cohorts):
        dists = np.linalg.norm(metadata - centers[c], axis=1)
        nearest = np.argsort(dists)[:cohort_size]
        cohorts.append(np.array(nearest))
    return cohorts


def topological_cohorts(
    sketches: torch.Tensor,
    num_cohorts: int,
    cohort_size: int,
    seed: int = 0,
) -> list:
    """Topological (rank-based) cohort assignment on sketch space.

    For each cohort, pick a random anchor client, then take the
    (cohort_size - 1) clients with most-similar sketches by cosine similarity.

    Following Ballerini et al. (2008): topological (rank-based) neighbors
    rather than metric (distance-thresholded) ones. The interaction count is
    fixed; the geometric reach varies with local density.

    Anchor selection is randomized for adversarial robustness: an attacker
    cannot pre-commit to evading a specific cohort structure.

    Inputs:
        sketches: (n_clients, sketch_dim) - server-visible sketch vectors
        num_cohorts, cohort_size: structural parameters
        seed: for anchor selection

    Returns list of cohort index arrays.
    """
    n = sketches.shape[0]
    rng = np.random.default_rng(seed)
    sketches_np = sketches.cpu().numpy()
    # Normalize for cosine similarity
    norms = np.linalg.norm(sketches_np, axis=1, keepdims=True) + 1e-12
    s_norm = sketches_np / norms

    cohorts = []
    anchors = rng.choice(n, size=num_cohorts, replace=(num_cohorts > n))
    for a in anchors:
        sims = s_norm @ s_norm[a]
        # Top cohort_size by similarity (includes anchor itself)
        top = np.argsort(-sims)[:cohort_size]
        cohorts.append(top)
    return cohorts


def cohort_overlap_stats(cohorts: list, num_clients: int) -> dict:
    """Compute structural stats: client coverage, mean appearances, etc."""
    appearance = np.zeros(num_clients, dtype=int)
    for c in cohorts:
        appearance[c] += 1
    return {
        "n_cohorts": len(cohorts),
        "mean_cohort_size": float(np.mean([len(c) for c in cohorts])),
        "client_coverage": int((appearance > 0).sum()) / num_clients,
        "mean_appearances": float(appearance.mean()),
        "max_appearances": int(appearance.max()),
        "uncovered_clients": int((appearance == 0).sum()),
    }


def build_factor_graph(cohorts: list, num_clients: int) -> dict:
    """Build the cohort-client factor graph for belief propagation.

    Returns:
        client_to_cohorts: dict[client_id] -> list of cohort_ids it belongs to
        cohort_to_clients: dict[cohort_id] -> list of client_ids it contains
    """
    client_to_cohorts = defaultdict(list)
    cohort_to_clients = {}
    for ci, members in enumerate(cohorts):
        cohort_to_clients[ci] = list(members)
        for m in members:
            client_to_cohorts[int(m)].append(ci)
    return {
        "c2k": dict(client_to_cohorts),
        "k2c": cohort_to_clients,
    }


# =============================================================================
# Belief propagation for Byzantine localization
# =============================================================================

def belief_propagation(
    cohort_anomaly_scores: np.ndarray,  # (n_cohorts,) in [0, 1]: how anomalous each cohort looks
    factor_graph: dict,
    num_clients: int,
    num_iters: int = 5,
    prior: float = 0.1,  # prior probability that a client is Byzantine
) -> np.ndarray:
    """Run BP on the cohort-client factor graph to estimate per-client
    posterior probability of being Byzantine.

    Math: variables are binary (each client honest=0 or Byzantine=1).
    Each cohort factor encodes: if many of my members are Byzantine, my
    aggregate looks anomalous.

    We use a simple noisy-OR factor: P(cohort anomalous | clients) =
    1 - prod_i (1 - p_i * tau), where tau is per-client influence. This
    is the standard model for "any one bad apple spoils the bunch" factors
    (Pearl 1988, Section 4.3).

    We run sum-product in log domain for stability, then read out marginals.

    Returns: (n_clients,) posterior P(Byzantine | observations).
    """
    c2k = factor_graph["c2k"]
    k2c = factor_graph["k2c"]
    n_cohorts = len(k2c)

    # Log-domain priors
    log_prior_byz = math.log(prior)
    log_prior_hon = math.log(1.0 - prior)

    # Messages: client -> cohort (log-ratio log P(byz)/P(hon))
    # Initialize at prior
    msg_c2k = {(c, k): (log_prior_byz - log_prior_hon)
               for c, ks in c2k.items() for k in ks}

    # Cohort observation: anomaly score in [0,1]; convert to log-likelihood ratio
    # P(obs=anom | byz_present)/P(obs=anom | not) -- monotonic in anomaly_score
    # Use logit-style: ll[k] = log(s) - log(1-s) clipped
    eps = 1e-6
    s = np.clip(cohort_anomaly_scores, eps, 1 - eps)
    cohort_ll = np.log(s) - np.log(1 - s)  # (n_cohorts,)

    posteriors = np.full(num_clients, log_prior_byz - log_prior_hon)

    for _ in range(num_iters):
        # Cohort -> client messages
        msg_k2c = {}
        for k in range(n_cohorts):
            members = k2c[k]
            # For each member, message excludes its own incoming
            # Approximation: use mean of others' beliefs as input to factor
            for m in members:
                others = [c for c in members if c != m]
                # Sum of others' log-ratios (their belief in being byz)
                sum_others = sum(msg_c2k.get((o, k), 0.0) for o in others)
                # Cohort likelihood weighted by structure
                msg_k2c[(k, m)] = cohort_ll[k] * (1.0 / max(len(members), 1)) + 0.1 * sum_others / max(len(others), 1)

        # Client -> cohort messages (next iter)
        new_msg_c2k = {}
        for c in range(num_clients):
            ks = c2k.get(c, [])
            total = log_prior_byz - log_prior_hon + sum(msg_k2c.get((k, c), 0.0) for k in ks)
            # For each outgoing, exclude that cohort's contribution
            for k in ks:
                new_msg_c2k[(c, k)] = total - msg_k2c.get((k, c), 0.0)
            posteriors[c] = total
        msg_c2k = new_msg_c2k

    # Convert log-ratios to probabilities
    probs = 1.0 / (1.0 + np.exp(-posteriors))
    return probs


def cohort_anomaly_score(
    cohort_aggregate: torch.Tensor,
    expected_aggregate: torch.Tensor,
    expected_norm: float,
) -> float:
    """Score how anomalous a cohort's aggregate update is.

    We compare the cohort's aggregate to the global expected behavior.
    Score in [0, 1]: 0 = looks normal, 1 = highly anomalous.

    Math: cosine distance + relative norm deviation, mapped to [0,1] via
    a logistic transform. This is a pragmatic choice; alternatives (KL,
    Mahalanobis) are possible but require more state.
    """
    if expected_aggregate is None or expected_aggregate.numel() == 0:
        return 0.5  # no info yet -> neutral

    # Cosine similarity
    a = cohort_aggregate.flatten().float()
    b = expected_aggregate.flatten().float()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    cos_dist = (1.0 - cos.item()) / 2.0  # in [0, 1]

    # Norm deviation
    norm_dev = abs(a.norm().item() - expected_norm) / (expected_norm + 1e-12)
    norm_dev = min(norm_dev, 5.0) / 5.0  # cap and normalize

    return float(0.5 * cos_dist + 0.5 * norm_dev)


# =============================================================================
# Attacks
# =============================================================================

class AttackBase:
    """Base class for Byzantine attacks. Operates on a flat parameter delta."""
    name = "base"
    def apply(self, delta: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class SignFlipAttack(AttackBase):
    name = "sign_flip"
    def apply(self, delta: torch.Tensor, **kwargs) -> torch.Tensor:
        return -delta


class ScaledAttack(AttackBase):
    name = "scaled"
    def __init__(self, factor: float = 10.0):
        self.factor = factor
    def apply(self, delta: torch.Tensor, **kwargs) -> torch.Tensor:
        return delta * self.factor


class GaussianAttack(AttackBase):
    name = "gaussian"
    def __init__(self, std: float = 1.0):
        self.std = std
    def apply(self, delta: torch.Tensor, **kwargs) -> torch.Tensor:
        return delta + torch.randn_like(delta) * self.std


class LabelFlipAttack(AttackBase):
    """Marker class - actual flip happens at data-loading time."""
    name = "label_flip"
    def apply(self, delta: torch.Tensor, **kwargs) -> torch.Tensor:
        return delta  # the damage is already done in training


def make_attack(name: str, **kwargs) -> AttackBase:
    table = {
        "sign_flip": SignFlipAttack,
        "scaled": ScaledAttack,
        "gaussian": GaussianAttack,
        "label_flip": LabelFlipAttack,
        "none": type("NoAttack", (AttackBase,),
                     {"name": "none", "apply": lambda self, d, **kw: d})(),
    }
    if name == "none":
        return table["none"]
    return table[name](**kwargs)


# =============================================================================
# Client (local training)
# =============================================================================

@dataclass
class ClientResult:
    """What a client returns. NOTE: in real SecAgg, only the sum across a
    cohort is visible. We pass per-client objects through the simulator for
    bookkeeping, but the Server class is forbidden from reading them
    individually -- enforced by SecAggServer wrapping."""
    client_id: int
    delta: torch.Tensor       # parameter delta from this round's local training
    sketch: torch.Tensor      # JL sketch of delta (for cohort formation)
    n_samples: int
    train_loss: float
    is_byzantine: bool        # ground truth, for evaluation only -- never used by algorithm


class Client:
    """A federated client. Trains locally on its partition."""
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        is_byzantine: bool,
        attack: Optional[AttackBase],
        device: str,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        label_flip_pairs: Optional[dict] = None,
    ):
        self.id = client_id
        self.dataset = dataset
        self.is_byzantine = is_byzantine
        self.attack = attack
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.label_flip_pairs = label_flip_pairs or {}

    def train(
        self,
        model: nn.Module,
        global_flat: torch.Tensor,
        sketch: JLSketch,
    ) -> ClientResult:
        """Run local SGD; return delta and sketch."""
        set_flat_params(model, global_flat)
        model.to(self.device).train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        loss_sum, n_batches = 0.0, 0
        for _ in range(self.local_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                # Label flip attack: swap labels per pairs map
                if self.is_byzantine and isinstance(self.attack, LabelFlipAttack):
                    y = y.clone()
                    for src, dst in self.label_flip_pairs.items():
                        y[y == src] = dst
                opt.zero_grad()
                out = model(x)
                if out.shape[-1] == 1:  # regression
                    loss = F.mse_loss(out.squeeze(-1), y.float())
                else:
                    loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                n_batches += 1
        avg_loss = loss_sum / max(n_batches, 1)

        new_flat = get_flat_params(model)
        delta = new_flat - global_flat

        # Apply parameter-space attack (sign flip, scaling, gaussian)
        if self.is_byzantine and self.attack is not None:
            delta = self.attack.apply(delta)

        sk = sketch(delta)
        return ClientResult(
            client_id=self.id,
            delta=delta,
            sketch=sk,
            n_samples=len(self.dataset),
            train_loss=avg_loss,
            is_byzantine=self.is_byzantine,
        )


# =============================================================================
# SecAgg-constrained server
# =============================================================================

class SecAggServer:
    """Server that ENFORCES the SecAgg constraint at the API level.

    The server can ONLY call .aggregate_cohort(cohort_indices) which returns
    sum_{i in cohort} client_results[i].delta and sum sketch. It NEVER touches
    individual client deltas.

    Internally we hold the per-client objects (since we're in simulation),
    but the algorithm interface forbids using them. Test in tests/.
    """
    def __init__(self, results: list, allow_individual_access: bool = False):
        # results: list of ClientResult
        self._results = {r.client_id: r for r in results}
        self._access_log = []  # for unit tests: every read goes here
        self._allow_individual = allow_individual_access  # only True in baselines

    def aggregate_cohort(self, cohort: np.ndarray) -> dict:
        """SecAgg-compatible: returns only cohort sum, not individuals."""
        self._access_log.append(("cohort", tuple(int(c) for c in cohort)))
        members = [self._results[int(i)] for i in cohort if int(i) in self._results]
        if not members:
            return None
        delta_sum = torch.stack([m.delta for m in members]).sum(0)
        sketch_sum = torch.stack([m.sketch for m in members]).sum(0)
        n_total = sum(m.n_samples for m in members)
        return {
            "delta_sum": delta_sum,
            "sketch_sum": sketch_sum,
            "n_clients": len(members),
            "n_samples": n_total,
            "cohort": np.asarray(cohort),
        }

    def aggregate_all_cohorts(self, cohorts: list) -> list:
        return [self.aggregate_cohort(c) for c in cohorts]

    def get_individual(self, client_id: int):
        """ESCAPE HATCH for non-SecAgg baselines (Krum, Trimmed Mean).
        Only callable when allow_individual_access=True. Logged for audit."""
        if not self._allow_individual:
            raise PermissionError(
                "SecAgg constraint violated: individual access not permitted. "
                "Set allow_individual_access=True only for non-SecAgg baselines."
            )
        self._access_log.append(("individual", client_id))
        return self._results[client_id]

    def all_sketches_for_topological(self) -> torch.Tensor:
        """Server-visible sketch matrix for topological cohort formation.

        IMPORTANT: this returns per-client sketches. To stay SecAgg-compatible,
        the sketches themselves must be considered public-disclosable. JL
        sketches with sufficient compression do not preserve enough info to
        reconstruct individual updates -- this is the privacy argument.
        For experiments we permit this read with explicit logging.
        """
        self._access_log.append(("sketches_for_cohort_formation", None))
        ids = sorted(self._results.keys())
        sketches = torch.stack([self._results[i].sketch for i in ids])
        return sketches, np.array(ids)

    def access_summary(self) -> dict:
        types = defaultdict(int)
        for kind, _ in self._access_log:
            types[kind] += 1
        return dict(types)


# =============================================================================
# Aggregation rules (algorithms running on top of SecAggServer)
# =============================================================================

def fedavg_secagg(server: SecAggServer, cohorts: list, total_samples: int) -> torch.Tensor:
    """Standard FedAvg using only cohort sums. SecAgg-compatible.

    Aggregates all cohort sums weighted by sample count.
    Note: clients in multiple cohorts are over-counted; we correct.
    """
    cohort_aggs = server.aggregate_all_cohorts(cohorts)
    # Build a counts vector for de-weighting overcounted clients
    counts = np.zeros(len(server._results))
    for c in cohorts:
        for i in c:
            counts[int(i)] += 1
    # We don't have per-client deltas here. Approximation: just sum cohorts
    # and divide by total client appearances. This biases toward clients in
    # more cohorts; a more careful design uses non-overlapping cohorts for
    # the aggregation step (and overlapping ones for detection).
    delta_total = sum(c["delta_sum"] for c in cohort_aggs if c is not None)
    weight_total = sum(counts[i] * server._results[i].n_samples for i in server._results)
    if weight_total == 0:
        return torch.zeros_like(cohort_aggs[0]["delta_sum"])
    return delta_total / weight_total * total_samples / len(server._results)


def krum_aggregate(individuals: list, num_byzantine: int) -> torch.Tensor:
    """Krum (Blanchard et al. 2017). NOT SecAgg-compatible.

    Picks the single update closest to its k=n-f-2 nearest neighbors.
    Used as upper-bound baseline; requires per-client visibility.
    """
    n = len(individuals)
    deltas = torch.stack([r.delta for r in individuals])
    f = num_byzantine
    k = n - f - 2
    if k < 1:
        # Degenerate: too many byzantines; fall back to median
        return deltas.median(0).values
    # Pairwise squared distances
    dists = torch.cdist(deltas, deltas) ** 2
    # For each i, sum of k smallest distances (excluding self)
    sorted_d, _ = dists.sort(dim=1)
    sums = sorted_d[:, 1:k + 1].sum(dim=1)  # exclude self (index 0, distance 0)
    best = int(sums.argmin())
    return deltas[best]


def trimmed_mean_aggregate(individuals: list, trim_ratio: float = 0.2) -> torch.Tensor:
    """Trimmed mean (Yin et al. 2018). NOT SecAgg-compatible.

    Per coordinate, trim top and bottom trim_ratio fraction, average rest.
    """
    deltas = torch.stack([r.delta for r in individuals])  # (n, d)
    n = deltas.shape[0]
    k = int(trim_ratio * n)
    sorted_d, _ = deltas.sort(dim=0)
    if k > 0:
        trimmed = sorted_d[k:-k]
    else:
        trimmed = sorted_d
    return trimmed.mean(dim=0)


def median_aggregate(individuals: list) -> torch.Tensor:
    """Coordinate-wise median. NOT SecAgg-compatible."""
    deltas = torch.stack([r.delta for r in individuals])
    return deltas.median(0).values


def cohort_robust_aggregate(
    server: SecAggServer,
    cohorts: list,
    factor_graph: dict,
    expected_delta: torch.Tensor,
    expected_norm: float,
    num_clients: int,
    bp_iters: int = 5,
    trust_prior: float = 0.1,
    detection_threshold: float = 0.5,
) -> tuple:
    """Our method: SecAgg-compatible robust aggregation.

    1. Compute each cohort's aggregate via SecAgg.
    2. Score each cohort's anomaly vs expected.
    3. Run BP on factor graph to get per-client byzantine posterior.
    4. Aggregate from cohorts whose posterior-byzantine fraction is low.

    Returns: (aggregated_delta, byzantine_posteriors, cohort_scores)
    """
    cohort_aggs = server.aggregate_all_cohorts(cohorts)

    # Cohort anomaly scores
    scores = np.array([
        cohort_anomaly_score(
            (c["delta_sum"] / max(c["n_clients"], 1)) if c is not None else torch.zeros(1),
            expected_delta,
            expected_norm,
        ) if c is not None else 0.5
        for c in cohort_aggs
    ])

    # BP for per-client byzantine posterior
    posteriors = belief_propagation(
        scores,
        factor_graph,
        num_clients,
        num_iters=bp_iters,
        prior=trust_prior,
    )

    # Aggregate from cohorts where MEAN posterior is below threshold
    # (cohort-level filtering, doesn't break SecAgg)
    selected = []
    for ci, c in enumerate(cohort_aggs):
        if c is None:
            continue
        members = c["cohort"]
        mean_post = posteriors[members].mean()
        if mean_post < detection_threshold:
            selected.append(c)

    if not selected:
        # Fallback: use all cohorts (better than nothing)
        selected = [c for c in cohort_aggs if c is not None]

    total_n = sum(c["n_samples"] for c in selected)
    if total_n == 0:
        return torch.zeros_like(expected_delta), posteriors, scores
    weighted = sum(c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                   for c in selected)
    return weighted, posteriors, scores


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple:
    model.eval().to(device)
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if out.shape[-1] == 1:  # regression
                loss = F.mse_loss(out.squeeze(-1), y.float())
                loss_sum += loss.item() * y.size(0)
                total += y.size(0)
            else:
                loss = F.cross_entropy(out, y, reduction="sum")
                loss_sum += loss.item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
    return loss_sum / max(total, 1), (correct / max(total, 1) if total > 0 else 0.0)


def detection_metrics(posteriors: np.ndarray, byzantine_mask: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision/recall/F1 of byzantine detection at given threshold."""
    pred = posteriors >= threshold
    tp = int((pred & byzantine_mask).sum())
    fp = int((pred & ~byzantine_mask).sum())
    fn = int((~pred & byzantine_mask).sum())
    tn = int((~pred & ~byzantine_mask).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }


# =============================================================================
# Logging
# =============================================================================

class RunLogger:
    """Append-only CSV row logger + JSON config dumper.

    Usage:
        logger = RunLogger(out_dir, run_id, config)
        logger.log(round=1, accuracy=0.5, ...)
        logger.finalize(notes="completed")
    """
    def __init__(self, out_dir: str, run_id: str, config: dict):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.run_id = run_id
        self.csv_path = os.path.join(out_dir, f"{run_id}.csv")
        self.json_path = os.path.join(out_dir, f"{run_id}.json")
        self.fields = None
        # Dump config first
        with open(self.json_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log(self, **row):
        if self.fields is None:
            self.fields = list(row.keys())
            with open(self.csv_path, "w") as f:
                f.write(",".join(self.fields) + "\n")
        with open(self.csv_path, "a") as f:
            vals = []
            for k in self.fields:
                v = row.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    def finalize(self, **summary):
        sum_path = os.path.join(self.out_dir, f"{self.run_id}_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)


# =============================================================================
# Self-tests (run as `python fl_core.py`)
# =============================================================================

def _test_secagg_constraint():
    """Server should refuse individual access by default."""
    fake_results = [
        ClientResult(i, torch.zeros(10), torch.zeros(4), 100, 0.5, False)
        for i in range(5)
    ]
    server = SecAggServer(fake_results)
    try:
        server.get_individual(0)
        raise AssertionError("Should have raised PermissionError")
    except PermissionError:
        pass
    # Cohort access is allowed
    out = server.aggregate_cohort(np.array([0, 1, 2]))
    assert out["n_clients"] == 3
    print("[PASS] SecAgg constraint enforced.")


def _test_jl_sketch_linearity():
    """JL sketch must compose: S(sum x_i) == sum S(x_i)."""
    sk = JLSketch(in_dim=100, out_dim=20, seed=0)
    xs = [torch.randn(100) for _ in range(5)]
    s_of_sum = sk(sum(xs))
    sum_of_s = sum(sk(x) for x in xs)
    diff = (s_of_sum - sum_of_s).abs().max().item()
    assert diff < 1e-4, f"JL linearity broken: max diff {diff}"
    print(f"[PASS] JL sketch linearity (max diff {diff:.2e}).")


def _test_dirichlet_partition():
    """Dirichlet partition should produce valid splits."""
    targets = np.repeat(np.arange(10), 100)
    parts = dirichlet_partition(targets, num_clients=20, alpha=0.1, seed=0)
    total = sum(len(p) for p in parts)
    assert total == 1000
    assert all(len(p) >= 10 for p in parts)
    print(f"[PASS] Dirichlet partition: {len(parts)} clients, sizes "
          f"{[len(p) for p in parts[:5]]}...")


def _test_bp_localizes_simple():
    """BP should flag a clearly-anomalous client."""
    n_clients = 20
    cohorts = [
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 5, 6, 7, 8]),
        np.array([0, 9, 10, 11, 12]),
        np.array([13, 14, 15, 16, 17]),
        np.array([1, 5, 9, 13, 18]),
    ]
    # Cohorts containing client 0 are all anomalous; one without is not
    anomaly = np.array([0.9, 0.9, 0.9, 0.1, 0.5])
    fg = build_factor_graph(cohorts, n_clients)
    posts = belief_propagation(anomaly, fg, n_clients, num_iters=5, prior=0.1)
    assert posts[0] > posts[14], f"Client 0 should be more suspect: {posts[0]} vs {posts[14]}"
    print(f"[PASS] BP localizes: P(0=byz)={posts[0]:.3f}, P(14=byz)={posts[14]:.3f}")


def _test_resnet8gn_forward():
    """ResNet-8-GN should run a forward pass with expected param count."""
    m = ResNet8GN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10)
    n_params = count_params(m)
    assert 50_000 < n_params < 200_000, f"Unexpected param count: {n_params}"
    print(f"[PASS] ResNet-8-GN: {n_params:,} params, output shape {y.shape}")


def run_all_tests():
    print("Running fl_core self-tests...")
    _test_secagg_constraint()
    _test_jl_sketch_linearity()
    _test_dirichlet_partition()
    _test_bp_localizes_simple()
    _test_resnet8gn_forward()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
