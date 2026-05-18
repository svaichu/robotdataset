"""TED memmap building from AgiBotWorld-Beta TAR archives."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from robotdataset.agibot.loader import download_tar, find_tar_for_episode
from robotdataset.agibot.video_decoder import decode_mp4_from_tar
from robotdataset.oxe.memmap_builder import is_episode_cached
from robotdataset.oxe.utils import episode_to_ted_steps

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

_EPISODE_SENTINEL = "_steps.json"


def _skill_at_frame(action_config: List[Dict], frame_idx: int) -> str:
    """Return the action_text for the segment covering ``frame_idx``, or ''."""
    for seg in action_config:
        if seg.get("start_frame", 0) <= frame_idx < seg.get("end_frame", 0):
            return seg.get("action_text", "")
    return ""


def _build_one_agibot_episode(
    task_id: int,
    episode_info: Dict,
    global_id: int,
    episode_dir: Path,
    cameras: List[str],
    repo_files: List[str],
    cache_dir: Path,
) -> int:
    """Download the TAR, decode video frames, convert to TED, and memmap.

    Returns the number of steps written (0 if the episode is empty or missing).
    """
    episode_id = episode_info["episode_id"]
    task_name = episode_info.get("task_name", "")
    action_config = episode_info.get("label_info", {}).get("action_config", [])

    # ------------------------------------------------------------------
    # 1. Find and download the TAR that contains this episode.
    # ------------------------------------------------------------------
    repo_path = find_tar_for_episode(task_id, episode_id, repo_files)
    tar_path = download_tar(repo_path, cache_dir)

    # ------------------------------------------------------------------
    # 2. Decode each camera's video → (T, H, W, 3) uint8.
    #    Missing cameras are silently skipped.
    # ------------------------------------------------------------------
    camera_frames: Dict[str, np.ndarray] = {}
    for cam in cameras:
        member = f"{episode_id}/videos/{cam}.mp4"
        try:
            camera_frames[cam] = decode_mp4_from_tar(tar_path, member)
        except (KeyError, Exception):
            pass  # camera not present for this episode

    if not camera_frames:
        return 0

    # Align to the shortest video stream (avoids off-by-one between cameras).
    T = min(arr.shape[0] for arr in camera_frames.values())

    # ------------------------------------------------------------------
    # 3. Build OXE-style episode dict and convert to TED steps.
    # ------------------------------------------------------------------
    steps = []
    for t in range(T):
        obs: Dict = {cam: camera_frames[cam][t] for cam in camera_frames}
        obs["language_instruction"] = task_name
        steps.append(
            {
                "observation": obs,
                "action": np.zeros(1, dtype=np.float32),
                "reward": 0.0,
                "is_last": t == T - 1,
                "is_terminal": t == T - 1,
            }
        )

    ted_steps = episode_to_ted_steps({"steps": steps}, global_id, tf_tensor_types=())
    if not ted_steps:
        return 0

    # ------------------------------------------------------------------
    # 4. Memmap the episode.
    # ------------------------------------------------------------------
    episode_dir.mkdir(parents=True, exist_ok=True)
    td = torch.stack(ted_steps)
    td.memmap_(str(episode_dir))
    n_steps = len(td)
    (episode_dir / _EPISODE_SENTINEL).write_text(json.dumps({"n_steps": n_steps}))
    return n_steps


def build_missing_agibot_episodes(
    task_infos: Dict[int, List[Dict]],
    episode_map: Dict[int, Tuple[int, int]],
    episodes_dir: Path,
    missing: List[int],
    cameras: List[str],
    repo_files: List[str],
    cache_dir: Path,
) -> None:
    """Download and cache only the TAR data needed for ``missing`` global episodes.

    For each missing episode:
    - Only the TAR file that contains that specific episode is downloaded.
    - Video is decoded for the selected ``cameras`` only.
    - Already-cached episodes are skipped (safe to resume after interruption).

    Args:
        task_infos: ``{task_id: [episode_info_dict, ...]}``.
        episode_map: ``{global_id: (task_id, episode_id)}``.
        episodes_dir: Root dir for per-episode TED memmaps.
        missing: Global episode IDs not yet cached.
        cameras: Camera names to decode (e.g. ``["head_color", "hand_left_color"]``).
        repo_files: Full file listing from ``get_repo_files()``.
        cache_dir: Root cache dir for TAR downloads and HF metadata.
    """
    if not missing:
        return

    # Build a flat lookup: (task_id, episode_id) → episode_info dict.
    ep_info_lookup: Dict[Tuple[int, int], Dict] = {
        (task_id, ep["episode_id"]): ep
        for task_id, episodes in task_infos.items()
        for ep in episodes
    }

    pbar = (
        _tqdm_cls(
            total=len(missing),
            desc="Converting AgiBot episodes to TED",
            unit="ep",
            dynamic_ncols=True,
        )
        if _tqdm_cls is not None
        else None
    )

    for gid in sorted(missing):
        task_id, episode_id = episode_map[gid]
        episode_dir = episodes_dir / str(gid)
        if is_episode_cached(episode_dir):
            if pbar is not None:
                pbar.update(1)
            continue
        ep_info = ep_info_lookup.get((task_id, episode_id))
        if ep_info is not None:
            _build_one_agibot_episode(
                task_id, ep_info, gid, episode_dir, cameras, repo_files, cache_dir
            )
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
