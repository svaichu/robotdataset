"""``DatasetStatsMixin``: ``generate_stats()`` / ``statistics`` for TED memmap datasets."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from robotdataset.stats import (
    DEFAULT_SAMPLE_SIZE,
    STATS_VERSION,
    EpisodeStats,
    compute_episode_stats,
    leaf_schema,
    merge_episode_stats,
    signal_leaves,
    stream_moments,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class DatasetStatsMixin:
    """Adds dataset-wide statistics over the loaded episodes.

    Use with the mixin first: ``class X(DatasetStatsMixin, BaseDatasetExperienceReplay)``.
    The host class provides ``_loaded_indices`` and ``_episodes_dir()`` and calls
    :meth:`_ensure_episode_stats` right after its episode-build step.  Records
    are read from the per-episode memmap, so this is independent of the loader.
    """

    stats_sample_size: int = DEFAULT_SAMPLE_SIZE

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _ensure_episode_stats(self, force: bool = False) -> None:
        """Make sure every selected episode has a valid record; never rebuilds episodes."""
        from tensordict import TensorDict

        episodes_dir = self._episodes_dir()
        records: Dict[int, EpisodeStats] = getattr(self, "_episode_stats", {})
        failures: Dict[int, str] = {}
        todo: List[int] = []
        tds: Dict[int, Any] = {}
        for i in self._loaded_indices:
            ep_dir = episodes_dir / str(i)
            td = TensorDict.load_memmap(str(ep_dir))
            tds[i] = td
            rec = None if force else (records.get(i) or EpisodeStats.load(ep_dir))
            if rec is not None and not _matches(rec, td):
                rec = None
            if rec is None:
                todo.append(i)
            else:
                records[i] = rec
        bar = (
            _tqdm(total=len(todo), desc="Computing episode statistics", unit="ep", dynamic_ncols=True)
            if todo and _tqdm is not None
            else None
        )
        unwritable = False
        for i in todo:
            try:
                rec = compute_episode_stats(tds[i], sample_size=self.stats_sample_size, seed=i)
            except ValueError as e:
                records.pop(i, None)
                failures[i] = str(e)
                warnings.warn(f"No statistics for episode {i}: {e}", stacklevel=3)
            else:
                records[i] = rec
                try:
                    rec.save(episodes_dir / str(i))
                except OSError as e:
                    if not unwritable:
                        warnings.warn(
                            f"Cannot write statistics under {episodes_dir} ({e}); "
                            "keeping them in memory only.",
                            stacklevel=3,
                        )
                    unwritable = True
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        self._episode_stats = records
        self._stats_failures = failures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_stats(
        self,
        keys: Optional[Sequence[str]] = None,
        force: bool = False,
        exact: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Normalization statistics over the loaded episodes.

        Args:
            keys: Slash-separated paths to include (default: every float signal
                leaf -- ``next/``, ``collector/``, flags, images and non-float
                leaves are always excluded).
            force: Recompute the per-episode records instead of reusing them.
            exact: Rescan the combined memmap for exact ``q01``/``q99``
                instead of merging per-episode samples.  Nothing is written.

        Returns:
            ``{path: {"mean", "std", "min", "max", "q01", "q99": list[float],
            "count": int}}``.

        Raises:
            ValueError: Unknown ``keys`` (lists the valid keys), or an episode
                has no valid record (non-finite data), or non-finite data when
                ``exact=True``.
        """
        if exact:
            return self._generate_exact_stats(keys)
        if force or not self._have_all_records():
            self._ensure_episode_stats(force=force)
        if self._stats_failures:
            detail = "; ".join(f"episode {i}: {m}" for i, m in sorted(self._stats_failures.items()))
            raise ValueError(f"Statistics unavailable for some episodes: {detail}")
        return merge_episode_stats([self._episode_stats[i] for i in self._loaded_indices], keys=keys)

    @property
    def statistics(self) -> Dict[str, Dict[str, Any]]:
        """Same as :meth:`generate_stats` with default arguments."""
        return self.generate_stats()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _have_all_records(self) -> bool:
        recs = getattr(self, "_episode_stats", {})
        return not getattr(self, "_stats_failures", {}) and all(
            i in recs for i in self._loaded_indices
        )

    def _generate_exact_stats(self, keys: Optional[Sequence[str]]) -> Dict[str, Dict[str, Any]]:
        """Streaming moments plus per-dimension ``linear`` quantiles of the combined memmap.

        Peak memory is ``rows * 8`` bytes for one dimension's column, plus one
        chunk of ``_CHUNK_ROWS`` rows of the feature being scanned.
        """
        td = self._storage._storage
        leaves = signal_leaves(td, keys)
        out: Dict[str, Dict[str, Any]] = {}
        dims = sum(leaf_schema(leaves).values())
        bar = (
            _tqdm(total=dims, desc="Exact quantiles", unit="dim", dynamic_ncols=True)
            if _tqdm is not None
            else None
        )
        for path, leaf in leaves.items():
            n = leaf.shape[0]
            acc = stream_moments(leaf, path)
            flat = leaf.reshape(n, -1)
            q01, q99 = [], []
            for j in range(flat.shape[1]):
                col = flat[:, j].to(torch.float64).numpy()
                lo, hi = np.quantile(col, [0.01, 0.99], method="linear")
                q01.append(float(lo))
                q99.append(float(hi))
                if bar is not None:
                    bar.update(1)
            out[path] = {
                "mean": acc.mean.tolist(),
                "std": np.sqrt(acc.m2 / acc.count).tolist(),
                "min": acc.min.tolist(),
                "max": acc.max.tolist(),
                "q01": q01,
                "q99": q99,
                "count": int(acc.count),
            }
        if bar is not None:
            bar.close()
        return out


def _matches(rec: EpisodeStats, td: Any) -> bool:
    """A record is valid only if row count and schema still match the episode."""
    return (
        rec.version == STATS_VERSION
        and rec.num_rows == len(td)
        and rec.schema == leaf_schema(signal_leaves(td))
    )
