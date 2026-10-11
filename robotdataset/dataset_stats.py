"""``DatasetStatsMixin``: ``generate_stats()`` / ``statistics`` for TED memmap datasets."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from robotdataset.relative_stats import (
    build_specs,
    compute_relative_episode_stats,
    episode_chunks,
    exact_relative_stats,
    merge_relative_stats,
)
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

    def generate_relative_action_stats(
        self,
        groups: Mapping[str, Tuple[int, int]],
        state_keys: Mapping[str, str],
        horizon: int,
        representation: Union[str, Mapping[str, str]] = "joint",
        action_key: str = "action",
        force: bool = False,
        exact: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Statistics of actions expressed relative to the current state.

        Every chunk of ``horizon`` consecutive actions is expressed relative to the
        state at its first step (see :mod:`robotdataset.relative_stats` for the exact
        definition, which follows GR00T), and statistics are taken over all chunks of
        the loaded episodes, per horizon step and dimension.

        Args:
            groups: ``{group: (start, end)}`` columns of ``action_key``.
            state_keys: ``{group: path}`` of the reference state (same dimension as the group).
            horizon: Chunk length (GR00T: the action horizon, e.g. 16).
            representation: ``"joint"`` (difference) or ``"xyz_rotvec"`` (pose relative
                to the reference pose; 6 columns), one for all groups or ``{group: name}``.
            action_key: Path of the action tensor.
            force: Recompute the per-episode records instead of reusing them.
            exact: Rescan every chunk for exact ``q01``/``q99`` instead of merging
                per-episode samples.  Nothing is written; peak memory is about
                ``chunks * horizon * dim * 8`` bytes.

        Returns:
            ``{group: {"mean", "std", "min", "max", "q01", "q99": (horizon, dim) nested
            lists, "count": number of chunks}}``; pass it as ``relative_action=`` to
            :func:`robotdataset.gr00t.to_gr00t_statistics`.

        Raises:
            ValueError: Invalid groups/keys/dimensions, non-finite data, or all loaded
                episodes shorter than ``horizon``.
        """
        specs = build_specs(groups, state_keys, horizon, representation, action_key)
        out: Dict[str, Dict[str, Any]] = {}
        for spec in specs:
            if exact:
                out[spec.group] = self._exact_relative_stats(spec)
            else:
                records = self._ensure_relative_records(spec, force)
                out[spec.group] = merge_relative_stats(records, spec)
        return out

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

    def _ensure_relative_records(self, spec, force: bool) -> List[EpisodeStats]:
        """Records of the loaded episodes for one relative group (computed once, then cached)."""
        from tensordict import TensorDict

        episodes_dir = self._episodes_dir()
        by_spec: Dict[str, Dict[int, Optional[EpisodeStats]]] = self.__dict__.setdefault(
            "_relative_records", {}
        )
        cache = by_spec.setdefault(spec.stem, {})
        todo: List[Tuple[int, Any]] = []
        for i in self._loaded_indices:
            td = TensorDict.load_memmap(str(episodes_dir / str(i)))
            expected = spec.expected_rows(len(td))
            if expected == 0:  # shorter than the horizon: no chunks, nothing to store
                cache[i] = None
                continue
            rec = None if force else (cache.get(i) or EpisodeStats.load(episodes_dir / str(i), spec.stem))
            if rec is not None and not (
                rec.num_rows == expected and rec.schema == {spec.path: spec.flat_dim}
            ):
                rec = None
            if rec is None:
                todo.append((i, td))
            else:
                cache[i] = rec
        bar = (
            _tqdm(total=len(todo), desc=f"Relative stats [{spec.group}]", unit="ep", dynamic_ncols=True)
            if todo and _tqdm is not None
            else None
        )
        unwritable = False
        for i, td in todo:
            try:
                rec = compute_relative_episode_stats(td, spec, sample_size=self.stats_sample_size, seed=i)
            except ValueError as e:
                raise ValueError(f"Relative statistics of group {spec.group!r}, episode {i}: {e}") from e
            cache[i] = rec
            try:
                rec.save(episodes_dir / str(i), spec.stem)
            except OSError as e:
                if not unwritable:
                    warnings.warn(
                        f"Cannot write statistics under {episodes_dir} ({e}); keeping them in memory only.",
                        stacklevel=3,
                    )
                unwritable = True
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        records = [cache[i] for i in self._loaded_indices if cache.get(i) is not None]
        if not records:
            raise ValueError(
                f"No relative-action chunks for group {spec.group!r}: every loaded episode has "
                f"fewer than horizon={spec.horizon} rows"
            )
        return records

    def _exact_relative_stats(self, spec) -> Dict[str, Any]:
        from tensordict import TensorDict

        episodes_dir = self._episodes_dir()
        chunks = [
            episode_chunks(TensorDict.load_memmap(str(episodes_dir / str(i))), spec)
            for i in self._loaded_indices
        ]
        return exact_relative_stats(np.concatenate(chunks, axis=0), spec)

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
