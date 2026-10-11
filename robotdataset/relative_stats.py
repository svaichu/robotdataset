"""Relative-action statistics: per-episode records over *action chunks*.

A policy that predicts actions relative to the current state needs normalization
statistics of those differences; they cannot be derived from the absolute-action
statistics.  This module follows the definition used by GR00T
(``gr00t/data/stats.py``, ``RelativeActionLoader``) without importing it:

- For an episode of ``n`` rows and a horizon ``H`` the chunk starts are
  ``i = 0 .. n - H`` (``n - H + 1`` chunks; chunks never run past the episode end and
  are never padded).  Episodes shorter than ``H`` contribute nothing.
- Chunk ``i`` holds the action rows ``i .. i + H - 1`` of the group's columns and is
  expressed relative to the reference state ``state[i]`` (the state at the first
  action step), so the result has shape ``(H, D)``.
- ``representation="joint"``: ``action - reference`` per dimension.
- ``representation="xyz_rotvec"`` (``D == 6``: translation + rotation vector): the pose
  ``T_ref^-1 @ T_action`` of each chunk row in the reference frame, returned as
  translation + rotation vector.
- Statistics are taken over all chunks of all episodes, per horizon step and dimension
  (shape ``(H, D)``), with the population std (``np.std``).

Records reuse :class:`robotdataset.stats.EpisodeStats` (exact moments plus a seeded
uniform sample of chunks) stored next to the absolute record as
``stats_rel_<fingerprint>.json`` / ``stats_rel_<fingerprint>_sample.npz``, so the
merge is the one used for absolute statistics: ``mean``/``std``/``min``/``max`` are
exact, ``q01``/``q99`` are estimated from the samples (``exact=True`` on the dataset
method rescans every chunk instead).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from robotdataset.stats import (
    DEFAULT_SAMPLE_SIZE,
    STATS_VERSION,
    EpisodeStats,
    MomentAccumulator,
    _check_finite,
    merge_episode_stats,
    signal_leaves,
)

REPRESENTATIONS = ("joint", "xyz_rotvec")
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelativeSpec:
    """One relative-action group.

    Attributes:
        group: Name of the group (e.g. ``"eef"``); the key of the result.
        action_slice: ``(start, end)`` columns of ``action_key`` that form the group.
        state_key: Slash-separated path of the reference state, with the same
            dimension as the group.
        representation: ``"joint"`` or ``"xyz_rotvec"``.
        horizon: Number of consecutive action steps per chunk.
        action_key: Slash-separated path of the action tensor.
    """

    group: str
    action_slice: Tuple[int, int]
    state_key: str
    representation: str
    horizon: int
    action_key: str = "action"

    def __post_init__(self) -> None:
        start, end = self.action_slice
        if not (0 <= start < end):
            raise ValueError(f"group {self.group!r}: invalid action range {self.action_slice}")
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"group {self.group!r}: representation {self.representation!r} is not one of "
                f"{REPRESENTATIONS}"
            )
        if self.representation == "xyz_rotvec" and end - start != 6:
            raise ValueError(
                f"group {self.group!r}: 'xyz_rotvec' needs 6 columns (xyz + rotvec), "
                f"got {end - start}"
            )
        if int(self.horizon) != self.horizon or self.horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {self.horizon!r}")

    @property
    def dim(self) -> int:
        return self.action_slice[1] - self.action_slice[0]

    @property
    def path(self) -> str:
        """Key of the record's single signal."""
        return f"relative_action/{self.group}"

    @property
    def flat_dim(self) -> int:
        return self.horizon * self.dim

    @property
    def stem(self) -> str:
        """File stem of the per-episode record; the fingerprint covers the whole spec."""
        payload = json.dumps(
            [self.group, list(self.action_slice), self.state_key, self.representation,
             int(self.horizon), self.action_key]
        )
        return "stats_rel_" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    def expected_rows(self, episode_len: int) -> int:
        """Number of chunks of an episode (``0`` if it is shorter than the horizon)."""
        return max(0, episode_len - self.horizon + 1)


def build_specs(
    groups: Mapping[str, Tuple[int, int]],
    state_keys: Mapping[str, str],
    horizon: int,
    representation: Union[str, Mapping[str, str]] = "joint",
    action_key: str = "action",
) -> List[RelativeSpec]:
    """Validate the arguments of ``generate_relative_action_stats`` and build specs."""
    if not groups:
        raise ValueError("groups must name at least one relative-action group")
    missing = [g for g in groups if g not in state_keys]
    if missing:
        raise ValueError(f"state_keys has no entry for group(s) {missing}")
    specs = []
    for g, (a, b) in groups.items():
        rep = representation if isinstance(representation, str) else representation.get(g, "joint")
        specs.append(
            RelativeSpec(g, (int(a), int(b)), state_keys[g], rep, int(horizon), action_key)
        )
    return specs


# ---------------------------------------------------------------------------
# Geometry (numpy, batched)
# ---------------------------------------------------------------------------

def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` from rotation vectors ``(..., 3)`` (Rodrigues)."""
    v = np.asarray(rotvec, dtype=np.float64)
    theta2 = np.sum(v * v, axis=-1)
    theta = np.sqrt(theta2)
    small = theta < 1e-4
    safe2 = np.where(small, 1.0, theta2)
    a = np.where(small, 1.0 - theta2 / 6.0, np.sin(theta) / np.where(small, 1.0, theta))
    b = np.where(small, 0.5 - theta2 / 24.0, (1.0 - np.cos(theta)) / safe2)
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -v[..., 2], v[..., 1]
    k[..., 1, 0], k[..., 1, 2] = v[..., 2], -v[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -v[..., 1], v[..., 0]
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def matrix_to_rotvec(rot: np.ndarray) -> np.ndarray:
    """Rotation vectors ``(..., 3)`` (angle in ``[0, pi]``) from rotation matrices ``(..., 3, 3)``."""
    m = np.asarray(rot, dtype=np.float64)
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    diag = np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], tr], axis=1)
    choice = np.argmax(diag, axis=1)  # largest of w^2, x^2, y^2, z^2 (Shepperd's method)
    quat = np.empty((m.shape[0], 4))  # x, y, z, w
    for c in range(4):
        sel = choice == c
        if not sel.any():
            continue
        r = m[sel]
        if c == 3:
            w = 0.5 * np.sqrt(1.0 + tr[sel])
            quat[sel] = np.stack(
                [(r[:, 2, 1] - r[:, 1, 2]) / (4 * w), (r[:, 0, 2] - r[:, 2, 0]) / (4 * w),
                 (r[:, 1, 0] - r[:, 0, 1]) / (4 * w), w], axis=1)
        else:
            i, j, k = c, (c + 1) % 3, (c + 2) % 3
            s = 2.0 * np.sqrt(1.0 + r[:, i, i] - r[:, j, j] - r[:, k, k])
            q = np.empty((r.shape[0], 4))
            q[:, i] = 0.25 * s
            q[:, j] = (r[:, j, i] + r[:, i, j]) / s
            q[:, k] = (r[:, k, i] + r[:, i, k]) / s
            q[:, 3] = (r[:, k, j] - r[:, j, k]) / s
            quat[sel] = q
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    quat[quat[:, 3] < 0] *= -1.0
    xyz, w = quat[:, :3], quat[:, 3]
    s = np.linalg.norm(xyz, axis=1)
    small = s < 1e-3
    safe = np.where(small, 1.0, s)
    scale = np.where(small, 2.0 / w * (1.0 - s**2 / (3.0 * w**2)), 2.0 * np.arctan2(s, w) / safe)
    return (xyz * scale[:, None]).reshape(batch + (3,))


def relative_chunks(actions: np.ndarray, states: np.ndarray, spec: RelativeSpec) -> np.ndarray:
    """All relative chunks of one episode as ``(n_chunks, horizon, D)`` ``float64``.

    Args:
        actions: ``(n, D_action)`` action columns of the episode.
        states: ``(n, D)`` reference state, same dimension as the group.
        spec: Group definition (horizon, representation, column range).

    Returns:
        ``(n - horizon + 1, horizon, D)``; zero chunks if the episode is shorter
        than the horizon.
    """
    start, end = spec.action_slice
    a = np.asarray(actions, dtype=np.float64)[:, start:end]
    ref = np.asarray(states, dtype=np.float64)
    n, h = a.shape[0], spec.horizon
    m = spec.expected_rows(n)
    if m == 0:
        return np.empty((0, h, spec.dim))
    window = np.arange(m)[:, None] + np.arange(h)[None, :]  # (m, h) rows of each chunk
    chunk = a[window]  # (m, h, D)
    ref = ref[:m]
    if spec.representation == "joint":
        return chunk - ref[:, None, :]
    r_ref = rotvec_to_matrix(ref[:, 3:])  # (m, 3, 3)
    r_act = rotvec_to_matrix(chunk[..., 3:])  # (m, h, 3, 3)
    rel_t = np.einsum("mji,mhj->mhi", r_ref, chunk[..., :3] - ref[:, None, :3])  # R_ref^T (p - p_ref)
    rel_r = np.einsum("mji,mhjk->mhik", r_ref, r_act)  # R_ref^T R
    return np.concatenate([rel_t, matrix_to_rotvec(rel_r)], axis=-1)


# ---------------------------------------------------------------------------
# Per-episode record and merge
# ---------------------------------------------------------------------------

def episode_chunks(td: Any, spec: RelativeSpec) -> np.ndarray:
    """Relative chunks of one episode memmap, ``(n_chunks, horizon, D)``.

    Raises:
        ValueError: A key is missing or not a float tensor, or the state's dimension
            differs from the group's.
    """
    leaves = signal_leaves(td, [spec.action_key, spec.state_key])
    action = leaves[spec.action_key]
    state = leaves[spec.state_key]
    if action.ndim != 2 or action.shape[1] < spec.action_slice[1]:
        raise ValueError(
            f"{spec.action_key!r} has shape {tuple(action.shape)}; group {spec.group!r} needs "
            f"columns {spec.action_slice} of a (rows, dim) tensor"
        )
    if state.ndim != 2 or state.shape[1] != spec.dim:
        raise ValueError(
            f"group {spec.group!r}: state {spec.state_key!r} has shape {tuple(state.shape)}, "
            f"expected (rows, {spec.dim}) to match the action columns {spec.action_slice}"
        )
    return relative_chunks(action.to(torch.float64).numpy(), state.to(torch.float64).numpy(), spec)


def compute_relative_episode_stats(
    td: Any,
    spec: RelativeSpec,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> Optional[EpisodeStats]:
    """Record of one episode, or ``None`` if the episode is shorter than the horizon.

    ``num_rows`` of the record is the number of chunks (the merge weights by it);
    the single signal ``spec.path`` has ``horizon * D`` columns (a chunk flattened
    row-major, horizon step first).

    Raises:
        ValueError: Non-finite values, or an invalid key / dimension.
    """
    chunks = episode_chunks(td, spec)
    m = chunks.shape[0]
    if m == 0:
        return None
    rows = chunks.reshape(m, spec.flat_dim)
    bad: List[int] = []
    _check_finite(rows, spec.path, bad)
    if bad:
        raise ValueError(
            f"Relative chunks of group {spec.group!r} have {sum(bad)} non-finite row(s) (NaN/inf)"
        )
    acc = MomentAccumulator(spec.flat_dim)
    acc.update(rows)
    k = min(sample_size, m)
    idx = np.sort(np.random.default_rng(seed).choice(m, size=k, replace=False))
    return EpisodeStats(
        STATS_VERSION, m, {spec.path: spec.flat_dim}, {spec.path: acc.as_dict()},
        {spec.path: rows[idx].copy()},
    )


def _shape(flat: Sequence[float], spec: RelativeSpec) -> List[List[float]]:
    return np.asarray(flat, dtype=np.float64).reshape(spec.horizon, spec.dim).tolist()


def merge_relative_stats(records: Sequence[EpisodeStats], spec: RelativeSpec) -> Dict[str, Any]:
    """Merge episode records into ``{mean, std, min, max, q01, q99: (H, D) lists, count}``.

    ``count`` is the number of chunks.
    """
    merged = merge_episode_stats(list(records), keys=[spec.path])[spec.path]
    out: Dict[str, Any] = {k: _shape(merged[k], spec) for k in _STAT_KEYS}
    out["count"] = merged["count"]
    return out


def exact_relative_stats(chunks: np.ndarray, spec: RelativeSpec) -> Dict[str, Any]:
    """Statistics of all chunks ``(N, horizon, D)`` at once (exact ``q01`` / ``q99``)."""
    n = chunks.shape[0]
    if n == 0:
        raise ValueError(f"No chunks for group {spec.group!r}: every episode is shorter than the horizon")
    flat = chunks.reshape(n, spec.flat_dim)
    if not np.isfinite(flat).all():
        raise ValueError(f"Relative chunks of group {spec.group!r} contain non-finite values (NaN/inf)")
    q = np.quantile(flat, [0.01, 0.99], axis=0)  # numpy's default 'linear' method
    stats = {
        "mean": flat.mean(axis=0), "std": flat.std(axis=0), "min": flat.min(axis=0),
        "max": flat.max(axis=0), "q01": q[0], "q99": q[1],
    }
    out: Dict[str, Any] = {k: _shape(v, spec) for k, v in stats.items()}
    out["count"] = int(n)
    return out
