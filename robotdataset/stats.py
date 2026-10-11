"""Per-episode normalization statistics for TED memmap episodes.

Statistics are computed once per episode, stored next to the episode, and
merged on demand for whatever episode selection a dataset loaded.

- ``count``, ``mean``, ``std``, ``min``, ``max`` are exact: each episode keeps
  streaming moments (count, mean, M2) that merge with the parallel
  (Chan et al.) update.  ``std`` is the population std (``np.std``).
- ``q01`` / ``q99`` are *estimates*: each episode keeps a seeded uniform row
  sample, and the merge takes a weighted quantile of the pooled samples.  Each
  sampled row of episode *i* carries weight ``n_i / k_i`` (rows in the episode
  over rows sampled).  For equal weights this reduces exactly to numpy's
  default ``linear`` quantile, so with ``sample_size >= episode length`` the
  result equals ``np.quantile`` over the concatenated data.  Exact quantiles
  for a loaded dataset are available through ``generate_stats(exact=True)``.

Keys are slash-separated paths (``"action"``, ``"observation/state/joints"``),
the same convention as ``delta_timestamps``.

On disk, next to each episode's memmap::

    stats.json         version, num_rows, schema, moments
    stats_sample.npz   sampled rows per path (url-quoted path as array name)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

import numpy as np
import torch

STATS_VERSION = 1
STATS_FILE = "stats.json"
SAMPLE_FILE = "stats_sample.npz"
DEFAULT_SAMPLE_SIZE = 256

_CHUNK_ROWS = 65536
_EXCLUDED_ROOTS = ("next", "collector")
_FLAG_NAMES = frozenset(
    {"done", "terminated", "truncated", "reward", "is_first", "is_last", "is_terminal"}
)
_IMAGE_HINTS = ("image", "pixel", "video", "rgb", "depth")


# ---------------------------------------------------------------------------
# Leaf selection
# ---------------------------------------------------------------------------

def _is_float_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor) and x.dtype.is_floating_point and x.ndim >= 1


def _is_default_signal(parts: Tuple[str, ...], leaf: Any) -> bool:
    if parts[0] in _EXCLUDED_ROOTS or parts[-1] in _FLAG_NAMES:
        return False
    if any(h in p.lower() for p in parts for h in _IMAGE_HINTS):
        return False
    return _is_float_tensor(leaf)


def signal_leaves(td: Any, keys: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
    """Return ``{path: tensor}`` for the leaves statistics are computed on.

    ``keys=None`` selects float signal leaves, excluding ``next/``,
    ``collector/``, flags, image-like paths and non-float dtypes.  Explicit
    ``keys`` may name any float tensor leaf.

    Raises:
        ValueError: A requested key is missing or not a float tensor; the
            message lists the valid default keys.
    """
    all_leaves = {
        ("/".join(k) if isinstance(k, tuple) else k): td.get(k)
        for k in td.keys(include_nested=True, leaves_only=True)
    }
    default = {
        p: v for p, v in all_leaves.items() if _is_default_signal(tuple(p.split("/")), v)
    }
    if keys is None:
        return default
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        leaf = all_leaves.get(k)
        if leaf is None or not _is_float_tensor(leaf):
            raise ValueError(
                f"Cannot compute statistics for key {k!r}; valid keys: {sorted(default)}"
            )
        out[k] = leaf
    return out


def leaf_schema(leaves: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """``{path: D}`` with ``D = prod(shape[1:])`` (trailing dims are flattened)."""
    return {p: int(np.prod(v.shape[1:], dtype=np.int64)) for p, v in leaves.items()}


def _iter_chunks(leaf: torch.Tensor, n: int, d: int) -> Iterable[np.ndarray]:
    flat = leaf.reshape(n, d)
    for s in range(0, n, _CHUNK_ROWS):
        yield flat[s : s + _CHUNK_ROWS].to(torch.float64).numpy()


class MomentAccumulator:
    """Streaming count / mean / M2 / min / max over rows of a ``(n, D)`` stream."""

    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.min = np.full(dim, np.inf)
        self.max = np.full(dim, -np.inf)

    def update(self, rows: np.ndarray) -> None:
        n = rows.shape[0]
        if n == 0:
            return
        c_mean = rows.mean(axis=0)
        c_m2 = ((rows - c_mean) ** 2).sum(axis=0)
        self.merge(n, c_mean, c_m2, rows.min(axis=0), rows.max(axis=0))

    def merge(self, n, mean, m2, mn, mx) -> None:
        total = self.count + n
        delta = mean - self.mean
        self.m2 = self.m2 + m2 + delta**2 * (self.count * n / total)
        self.mean = self.mean + delta * (n / total)
        self.count = total
        self.min = np.minimum(self.min, mn)
        self.max = np.maximum(self.max, mx)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


def _check_finite(rows: np.ndarray, path: str, bad: List[int]) -> None:
    mask = ~np.isfinite(rows).all(axis=1)
    if mask.any():
        bad.append(int(mask.sum()))


def stream_moments(leaf: torch.Tensor, path: str) -> MomentAccumulator:
    """One streaming pass over ``leaf``; raises ``ValueError`` on non-finite rows."""
    n = leaf.shape[0]
    d = int(np.prod(leaf.shape[1:], dtype=np.int64))
    acc = MomentAccumulator(d)
    bad: List[int] = []
    for rows in _iter_chunks(leaf, n, d):
        _check_finite(rows, path, bad)
        if not bad:
            acc.update(rows)
    if bad:
        raise ValueError(
            f"Feature {path!r} has {sum(bad)} row(s) with non-finite values (NaN/inf)"
        )
    return acc


# ---------------------------------------------------------------------------
# EpisodeStats record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Statistics record of one episode (see module docstring)."""

    version: int
    num_rows: int
    schema: Dict[str, int]
    moments: Dict[str, Dict[str, Any]]
    sample: Dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, episode_dir: Path | str, stem: str = "stats") -> None:
        """Atomically write ``<stem>.json`` and ``<stem>_sample.npz`` into ``episode_dir``.

        The sample file is written first; the json file is the validity marker.
        ``stem`` is ``"stats"`` for the absolute-signal record.
        """
        d = Path(episode_dir)
        arrays = {quote(p, safe=""): a for p, a in self.sample.items()}
        _atomic_write(d / f"{stem}_sample.npz", lambda f: np.savez(f, **arrays))
        payload = {
            "version": self.version,
            "num_rows": self.num_rows,
            "schema": self.schema,
            "moments": self.moments,
        }
        _atomic_write(d / f"{stem}.json", lambda f: f.write(json.dumps(payload).encode()))

    @classmethod
    def load(cls, episode_dir: Path | str, stem: str = "stats") -> Optional["EpisodeStats"]:
        """Load a record; ``None`` if missing, corrupted or of an unknown version."""
        d = Path(episode_dir)
        try:
            meta = json.loads((d / f"{stem}.json").read_text())
            if meta["version"] != STATS_VERSION:
                return None
            with np.load(d / f"{stem}_sample.npz") as npz:
                sample = {unquote(k): npz[k] for k in npz.files}
            rec = cls(
                version=int(meta["version"]),
                num_rows=int(meta["num_rows"]),
                schema={k: int(v) for k, v in meta["schema"].items()},
                moments=meta["moments"],
                sample=sample,
            )
        except Exception:
            return None
        if set(rec.sample) != set(rec.schema) or set(rec.moments) != set(rec.schema):
            return None
        for p, dim in rec.schema.items():
            if rec.sample[p].ndim != 2 or rec.sample[p].shape[1] != dim:
                return None
        return rec


def _atomic_write(path: Path, write) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def compute_episode_stats(
    td: Any,
    keys: Optional[Sequence[str]] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> EpisodeStats:
    """Compute the statistics record of one episode.

    Args:
        td: The episode's TensorDict, e.g. ``TensorDict.load_memmap(dir)``.
        keys: Slash-separated paths; ``None`` selects the default signal leaves.
        sample_size: Rows kept for quantile estimation (all rows if the episode
            is shorter).  Sampled uniformly without replacement.
        seed: RNG seed; datasets pass the episode id.

    Raises:
        ValueError: Non-finite values (names the feature and bad row count), or
            an invalid key.
    """
    leaves = signal_leaves(td, keys)
    n = len(td)
    if n == 0:
        raise ValueError("Cannot compute statistics of an empty episode")
    rng = np.random.default_rng(seed)
    k = min(sample_size, n)
    idx = np.sort(rng.choice(n, size=k, replace=False))

    moments: Dict[str, Dict[str, Any]] = {}
    sample: Dict[str, np.ndarray] = {}
    for path, leaf in leaves.items():
        moments[path] = stream_moments(leaf, path).as_dict()
        d = int(np.prod(leaf.shape[1:], dtype=np.int64))
        sample[path] = (
            leaf.reshape(n, d)[torch.from_numpy(idx)].to(torch.float64).numpy().copy()
        )
    return EpisodeStats(STATS_VERSION, n, leaf_schema(leaves), moments, sample)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: Sequence[float]) -> np.ndarray:
    """Weighted quantiles of one dimension.

    Sorted point *j* sits at position ``h_j = (cumulative weight before j) /
    (cumulative weight before the last point)`` and quantiles interpolate
    linearly over ``h``.  With equal weights ``h_j = j / (N - 1)``, which is
    numpy's ``linear`` method exactly.
    """
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    if len(v) == 1:
        return np.full(len(q), v[0])
    before = np.concatenate([[0.0], np.cumsum(w)[:-1]])
    h = before / before[-1]
    return np.interp(np.asarray(q, dtype=np.float64), h, v)


def merge_episode_stats(
    records: Sequence[EpisodeStats],
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Merge episode records into ``{path: {mean, std, min, max, q01, q99, count}}``.

    Raises:
        ValueError: No records, an unknown key (lists the valid keys), or
            records whose schemas differ for the requested keys.
    """
    if not records:
        raise ValueError("merge_episode_stats needs at least one record")
    valid = sorted(records[0].schema)
    keys = list(valid) if keys is None else list(keys)
    unknown = [k for k in keys if k not in records[0].schema]
    if unknown:
        raise ValueError(f"Unknown statistics keys {unknown}; valid keys: {valid}")
    diffs = []
    for i, r in enumerate(records):
        for k in keys:
            if r.schema.get(k) != records[0].schema[k]:
                diffs.append(
                    f"record {i}: {k!r} has dim {r.schema.get(k)}, expected {records[0].schema[k]}"
                )
    if diffs:
        raise ValueError("Episode statistics schemas differ: " + "; ".join(diffs))

    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        d = records[0].schema[k]
        acc = MomentAccumulator(d)
        for r in records:
            m = r.moments[k]
            acc.merge(
                m["count"], np.asarray(m["mean"]), np.asarray(m["m2"]),
                np.asarray(m["min"]), np.asarray(m["max"]),
            )
        vals = np.concatenate([r.sample[k] for r in records], axis=0)
        w = np.concatenate(
            [np.full(len(r.sample[k]), r.num_rows / len(r.sample[k])) for r in records]
        )
        q = np.stack(
            [weighted_quantile(vals[:, j], w, (0.01, 0.99)) for j in range(d)], axis=1
        )
        out[k] = {
            "mean": acc.mean.tolist(),
            "std": np.sqrt(acc.m2 / acc.count).tolist(),
            "min": acc.min.tolist(),
            "max": acc.max.tolist(),
            "q01": q[0].tolist(),
            "q99": q[1].tolist(),
            "count": int(acc.count),
        }
    return out
