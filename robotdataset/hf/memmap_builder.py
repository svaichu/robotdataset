from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

from robotdataset.hf.loader import (
    _maybe_episode_id_column,
    hf_episode_to_oxe_format,
    hf_row_to_oxe_episode,
)
from robotdataset.oxe.memmap_builder import is_episode_cached
from robotdataset.oxe.utils import episode_to_ted_steps

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

_EPISODE_SENTINEL = "_steps.json"


def _build_one_hf_episode(
    episode_dict: Dict[str, Any],
    episode_id: int,
    episode_dir: Path,
) -> int:
    """Convert one HF episode to TED steps and memmap it to ``episode_dir``.

    Returns the number of steps written.
    """
    steps = episode_to_ted_steps(episode_dict, episode_id, tf_tensor_types=())
    if not steps:
        return 0
    episode_dir.mkdir(parents=True, exist_ok=True)
    td = torch.stack(steps)
    td.memmap_(str(episode_dir))
    n_steps = len(td)
    (episode_dir / _EPISODE_SENTINEL).write_text(json.dumps({"n_steps": n_steps}))
    return n_steps


def build_missing_episodes(
    hf_dataset: Any,
    episodes_dir: Path,
    missing: List[int],
) -> None:
    """Convert and cache only the episodes in ``missing`` (episode IDs).

    Streams the HF dataset once, collecting rows for all missing episode IDs,
    then converts and memmaps each episode.  Already-cached episodes are
    skipped (safe to call if a previous run was interrupted).

    Args:
        hf_dataset: Loaded HuggingFace Dataset object.
        episodes_dir: Root directory for per-episode memmaps.
        missing: Episode IDs that are not yet cached.
    """
    if not missing:
        return

    missing_set: Set[int] = set(missing)
    col = _maybe_episode_id_column(hf_dataset.features)
    has_frame_index = "frame_index" in hf_dataset.features

    episode_rows: Dict[int, List[Any]] = {}
    if col is not None:
        # Single pass over row-per-step datasets: collect rows per missing episode ID.
        episode_rows = {eid: [] for eid in missing_set}
        for row in hf_dataset:
            eid = row[col]
            if eid in missing_set:
                episode_rows[eid].append(dict(row))

        if has_frame_index:
            for rows in episode_rows.values():
                rows.sort(key=lambda r: r["frame_index"])
    else:
        # Row-per-episode datasets: use row index as episode ID.
        for idx, row in enumerate(hf_dataset):
            if idx in missing_set:
                episode_rows[idx] = [dict(row)]

    pbar = (
        _tqdm_cls(
            total=len(missing_set),
            desc="Converting episodes to TED",
            unit="ep",
            dynamic_ncols=True,
        )
        if _tqdm_cls is not None
        else None
    )

    for episode_id in sorted(missing_set):
        episode_dir = episodes_dir / str(episode_id)
        if is_episode_cached(episode_dir):
            if pbar is not None:
                pbar.update(1)
            continue
        rows = episode_rows.get(episode_id, [])
        if rows:
            if col is not None:
                episode_dict = hf_episode_to_oxe_format(rows)
            else:
                episode_dict = hf_row_to_oxe_episode(rows[0])
            _build_one_hf_episode(episode_dict, episode_id, episode_dir)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
