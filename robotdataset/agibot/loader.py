"""HuggingFace loading utilities for AgiBotWorld-Beta."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HF_DATASET = "agibot-world/AgiBotWorld-Beta"

# Default cameras selected for decoding (subset that covers head + both wrists).
DEFAULT_CAMERAS: List[str] = [
    "head_color",
    "hand_left_color",
    "hand_right_color",
]

# All cameras present in the dataset (for reference / override).
ALL_CAMERAS: List[str] = [
    "head_color",
    "head_left_fisheye_color",
    "head_right_fisheye_color",
    "head_center_fisheye_color",
    "back_left_fisheye_color",
    "back_right_fisheye_color",
    "hand_left_color",
    "hand_right_color",
]


def _require_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        raise RuntimeError(
            "AgiBotWorldBetaDataset requires the 'huggingface_hub' package. "
            "Install it with: pip install 'robotdataset[hf]'"
        )


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def list_agibot_tasks() -> List[int]:
    """Return all available task IDs for AgiBotWorld-Beta.

    Task IDs are integers (e.g. 327, 351 …) matching the
    ``observations/{task_id}/`` directory structure in the HF repo.

    This is a metadata-only call — no observation data is downloaded.
    It reads the ``task_info/*.json`` file listing from the repo index.
    """
    hub = _require_hub()
    files = hub.list_repo_files(HF_DATASET, repo_type="dataset")
    task_ids: List[int] = []
    for f in files:
        if f.startswith("task_info/task_") and f.endswith(".json"):
            try:
                task_ids.append(int(f[len("task_info/task_"):-len(".json")]))
            except ValueError:
                pass
    return sorted(task_ids)


def get_repo_files() -> List[str]:
    """Return all file paths in the HuggingFace repo (one network call)."""
    hub = _require_hub()
    return list(hub.list_repo_files(HF_DATASET, repo_type="dataset"))


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

def load_task_info(task_id: int, cache_dir: Optional[Path] = None) -> List[Dict]:
    """Download ``task_info/task_{task_id}.json`` and return the episode list.

    The JSON is a list of episode dicts, each containing:
    - ``episode_id``: int
    - ``task_name``: str
    - ``init_scene_text``: str
    - ``label_info.action_config``: list of skill segments with
      ``start_frame``, ``end_frame``, ``action_text``, ``skill``

    Args:
        task_id: Integer task identifier.
        cache_dir: Root cache dir; HF file is stored under ``cache_dir/hf_cache``.
    """
    hub = _require_hub()
    kwargs: Dict = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir / "hf_cache")
    local = hub.hf_hub_download(
        repo_id=HF_DATASET,
        filename=f"task_info/task_{task_id}.json",
        repo_type="dataset",
        **kwargs,
    )
    return json.loads(Path(local).read_text())


# ---------------------------------------------------------------------------
# Episode → global ID mapping
# ---------------------------------------------------------------------------

def build_episode_mapping(
    task_infos: Dict[int, List[Dict]],
) -> Dict[int, Tuple[int, int]]:
    """Build a deterministic global episode ID mapping across tasks.

    Iterates tasks in ascending task_id order, and episodes within each task
    in ascending episode_id order.

    Args:
        task_infos: ``{task_id: [episode_info_dict, ...]}``.

    Returns:
        ``{global_id: (task_id, episode_id)}``.
    """
    mapping: Dict[int, Tuple[int, int]] = {}
    global_id = 0
    for task_id in sorted(task_infos):
        episodes = sorted(task_infos[task_id], key=lambda e: e["episode_id"])
        for ep in episodes:
            mapping[global_id] = (task_id, ep["episode_id"])
            global_id += 1
    return mapping


def save_episode_map(path: Path, mapping: Dict[int, Tuple[int, int]]) -> None:
    """Persist the episode mapping to a JSON file.

    Both keys and tuple values are serialised as strings/lists (JSON limitation);
    ``load_episode_map`` restores them to integers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(k): list(v) for k, v in mapping.items()}
    path.write_text(json.dumps(serialisable, indent=2))


def load_episode_map(path: Path) -> Dict[int, Tuple[int, int]]:
    """Load and restore the episode mapping from a JSON file."""
    raw = json.loads(path.read_text())
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# TAR file management
# ---------------------------------------------------------------------------

def find_tar_for_episode(
    task_id: int,
    episode_id: int,
    repo_files: List[str],
) -> str:
    """Return the repo-relative TAR path that contains ``episode_id``.

    TAR filenames encode the episode ID range: ``{start}-{end}.tar``.
    An episode belongs to the first TAR where ``start <= episode_id <= end``.

    Args:
        task_id: Integer task identifier.
        episode_id: The episode ID to locate.
        repo_files: Full list of paths from ``get_repo_files()``.

    Returns:
        Repo-relative path, e.g. ``"observations/327/648642-685046.tar"``.

    Raises:
        ValueError: If no TAR covers the given episode.
    """
    prefix = f"observations/{task_id}/"
    for f in repo_files:
        if not (f.startswith(prefix) and f.endswith(".tar")):
            continue
        fname = f[len(prefix):]  # "648642-685046.tar"
        try:
            start_str, rest = fname.split("-", 1)
            end_str = rest[: -len(".tar")]
            start, end = int(start_str), int(end_str)
        except ValueError:
            continue
        if start <= episode_id <= end:
            return f
    raise ValueError(
        f"No TAR found for task_id={task_id}, episode_id={episode_id}. "
        f"Checked prefix {prefix!r} in {len(repo_files)} repo files."
    )


def download_tar(repo_file_path: str, cache_dir: Path) -> Path:
    """Download a TAR file from HuggingFace and return its local path.

    Idempotent — HuggingFace Hub skips re-downloading cached files.

    Args:
        repo_file_path: Repo-relative path,
            e.g. ``"observations/327/648642-685046.tar"``.
        cache_dir: Root cache dir; file is stored under ``cache_dir/hf_cache``.
    """
    hub = _require_hub()
    local = hub.hf_hub_download(
        repo_id=HF_DATASET,
        filename=repo_file_path,
        repo_type="dataset",
        cache_dir=str(cache_dir / "hf_cache"),
    )
    return Path(local)
