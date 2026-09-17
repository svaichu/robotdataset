"""TED memmap building from LeRobot v2.1 parquet + mp4 assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from robotdataset.lerobot.loader import _META_COLUMNS, LeRobotSource
from robotdataset.lerobot.video_decoder import decode_mp4
from robotdataset.oxe.memmap_builder import is_episode_cached
from robotdataset.oxe.utils import episode_to_ted_steps

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:  # pragma: no cover - tqdm is a hard dependency in practice
    _tqdm_cls = None  # type: ignore[assignment]

_EPISODE_SENTINEL = "_steps.json"

#: Where a column that is both a leaf and a branch gets parked.  LeRobot
#: datasets commonly ship both ``observation.state`` (the concatenated vector)
#: and ``observation.state.joints`` (a slice of it); the nested TED layout
#: cannot hold a tensor and a sub-dict under one name, so the aggregate moves
#: to ``observation/state/all`` and the slices keep their own names.
AGGREGATE_LEAF = "all"


class FrameCountMismatch(ValueError):
    """A camera stream and the parquet disagree on how many steps an episode has.

    Almost always a frame-rate problem: some LeRobot exports carry an encoder
    ``video_info["video.fps"]`` that differs from the real frame rate, and a
    pipeline that resamples on the wrong one silently misaligns frames against
    states.  Raising here turns that into an error at build time instead of a
    policy that trains on shifted observations.
    """


def _to_leaf(value: Any) -> Any:
    """Normalise one parquet cell to a numpy array (or str for text)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value

    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    if array.dtype.kind in "fiub":
        array = array.astype(np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _assign_nested(tree: Dict[str, Any], parts: List[str], value: Any) -> None:
    """Insert ``value`` at the dotted path ``parts``, demoting leaf/branch clashes."""
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            child = node[part] = {AGGREGATE_LEAF: child}
        node = child

    leaf = parts[-1]
    existing = node.get(leaf)
    if isinstance(existing, dict):
        existing[AGGREGATE_LEAF] = value
    else:
        node[leaf] = value


def episode_to_steps(
    frame_table: Any,
    camera_frames: Dict[str, np.ndarray],
    task: Optional[str],
) -> List[Dict[str, Any]]:
    """Build OXE-style step dicts from one episode's parquet rows and frames.

    Column names are dot-separated in LeRobot and become nested dicts here, so
    ``observation.state.cartesian`` lands at ``observation/state/cartesian``.
    Decoded frames are attached at ``observation/images/<short_key>`` in HWC
    uint8 layout, matching what the rest of the package stores on disk, and the
    task string at ``observation/language_instruction``.

    Raises:
        FrameCountMismatch: If any camera's frame count differs from the
            number of parquet rows.
    """
    n_rows = len(frame_table)
    for key, frames in camera_frames.items():
        if frames.shape[0] != n_rows:
            raise FrameCountMismatch(
                f"{key}: decoded {frames.shape[0]} frames but the parquet has "
                f"{n_rows} rows. Check the dataset's frame rate -- meta/info.json "
                f"may advertise an encoder fps that differs from the real one."
            )

    columns = [c for c in frame_table.columns if c not in _META_COLUMNS]
    column_values = {c: frame_table[c].to_numpy() for c in columns}

    steps: List[Dict[str, Any]] = []
    for t in range(n_rows):
        step: Dict[str, Any] = {}
        for column in columns:
            _assign_nested(step, column.split("."), _to_leaf(column_values[column][t]))

        observation = step.get("observation")
        if not isinstance(observation, dict):
            observation = {AGGREGATE_LEAF: observation} if observation is not None else {}
            step["observation"] = observation

        if camera_frames:
            images = observation.setdefault("images", {})
            for key, frames in camera_frames.items():
                images[key.split(".")[-1]] = frames[t]
        if task is not None:
            observation["language_instruction"] = task

        step.setdefault("action", np.zeros(1, dtype=np.float32))
        step["reward"] = 0.0
        step["is_last"] = t == n_rows - 1
        step["is_terminal"] = t == n_rows - 1
        steps.append(step)

    return steps


def _build_one_episode(
    source: LeRobotSource,
    episode_index: int,
    episode_dir: Path,
    video_keys: List[str],
) -> int:
    """Fetch, decode, convert and memmap one episode.  Returns steps written."""
    import pandas as pd

    frame_table = pd.read_parquet(source.parquet_path(episode_index))
    camera_frames = {
        key: decode_mp4(source.video_path(episode_index, key)) for key in video_keys
    }

    steps = episode_to_steps(
        frame_table, camera_frames, source.task_for_episode(episode_index)
    )
    ted_steps = episode_to_ted_steps(
        {"steps": steps}, episode_index, tf_tensor_types=()
    )
    if not ted_steps:
        return 0

    episode_dir.mkdir(parents=True, exist_ok=True)
    td = torch.stack(ted_steps)
    td.memmap_(str(episode_dir))
    n_steps = len(td)
    (episode_dir / _EPISODE_SENTINEL).write_text(json.dumps({"n_steps": n_steps}))
    return n_steps


def build_missing_episodes(
    source: LeRobotSource,
    episodes_dir: Path,
    missing: List[int],
    video_keys: Optional[List[str]] = None,
) -> None:
    """Convert and cache only the episodes in ``missing`` as TED memmaps.

    Each episode is fetched, decoded and written independently, so peak memory
    is one episode's frames and an interrupted run resumes where it stopped.

    Args:
        source: Resolved :class:`~robotdataset.lerobot.loader.LeRobotSource`.
        episodes_dir: Root directory for per-episode memmaps.
        missing: Episode indices that are not yet cached.
        video_keys: Camera streams to decode.  Defaults to every video feature
            in ``meta/info.json``.
    """
    if not missing:
        return

    keys = list(source.video_keys if video_keys is None else video_keys)
    unknown = [k for k in keys if k not in source.video_keys]
    if unknown:
        raise KeyError(
            f"Unknown video keys {unknown}; available: {source.video_keys}"
        )

    pbar = (
        _tqdm_cls(
            total=len(missing),
            desc="Converting episodes to TED",
            unit="ep",
            dynamic_ncols=True,
        )
        if _tqdm_cls is not None
        else None
    )

    for episode_index in sorted(missing):
        episode_dir = episodes_dir / str(episode_index)
        if not is_episode_cached(episode_dir):
            _build_one_episode(source, episode_index, episode_dir, keys)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
