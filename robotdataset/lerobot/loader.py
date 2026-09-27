"""LeRobot v2.1 metadata parsing and selective asset fetching.

LeRobot datasets keep low-dimensional signals (state, action) in one parquet
file per episode and camera streams in standalone mp4 files, one per
``(episode, camera)`` pair::

    meta/info.json          templates, fps, feature schema
    meta/episodes.jsonl     per-episode length and task list
    meta/tasks.jsonl        task_index -> task string
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/<video_key>/episode_000000.mp4

:class:`LeRobotSource` resolves either origin -- a directory already on disk,
or a HuggingFace Hub repo id -- to concrete file paths, fetching only the
assets belonging to the episodes actually requested.  Loading two episodes of
a fifty-episode repo therefore transfers two episodes' worth of bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from robotdataset._common import _get_cache_dir

# Parquet columns that are dataset bookkeeping rather than observations.
_META_COLUMNS = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


def _require_hub() -> Any:
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError:
        raise RuntimeError(
            "Loading a LeRobot dataset by repo_id requires the 'huggingface_hub' "
            "package. Install it with: pip install 'robotdataset[hf]'"
        )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


class LeRobotSource:
    """Resolve LeRobot v2.1 assets from a local directory or the HuggingFace Hub.

    Exactly one of ``repo_id`` or ``local_path`` must be given.

    Args:
        repo_id: Hub dataset id, e.g. ``"LSY-lab/stack_without_ft_tact_v4"``.
            Files are fetched on demand into the robotdataset cache.
        local_path: Directory holding an already-materialised LeRobot tree
            (the layout produced by ``snapshot_download(..., local_dir=...)``
            or by ``lerobot.record``).  No network access is performed.
        cache_dir: Override the robotdataset cache root.  Resolution order is
            this argument, then ``ROBOTDATASET_CACHE``, then
            ``~/.cache/robotdataset``.
        revision: Hub revision (branch, tag or commit sha).  Pinning a commit
            sha makes a run reproducible across machines.

    Attributes:
        info: Parsed ``meta/info.json``.
        episodes_meta: ``{episode_index: episode_record}`` from
            ``meta/episodes.jsonl``.
        tasks: ``{task_index: task_string}`` from ``meta/tasks.jsonl``.
        root: Cache root for this dataset's TED memmaps and Hub downloads.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        if (repo_id is None) == (local_path is None):
            raise ValueError(
                "Provide exactly one of repo_id= or local_path= "
                f"(got repo_id={repo_id!r}, local_path={local_path!r})"
            )

        self.repo_id = repo_id
        self.revision = revision
        self.local_path = (
            Path(local_path).expanduser().resolve() if local_path is not None else None
        )
        self.slug = (
            repo_id.replace("/", "--") if repo_id is not None else self.local_path.name
        )
        self.root = _get_cache_dir(cache_dir) / "hf" / "lerobot_video" / self.slug
        self._hf_cache = self.root / "hf_cache"

        self.info: Dict[str, Any] = json.loads(self._fetch("meta/info.json").read_text())
        self.episodes_meta: Dict[int, Dict[str, Any]] = {
            int(e["episode_index"]): e
            for e in _read_jsonl(self._fetch("meta/episodes.jsonl"))
        }
        try:
            self.tasks: Dict[int, str] = {
                int(t["task_index"]): t["task"]
                for t in _read_jsonl(self._fetch("meta/tasks.jsonl"))
            }
        except Exception:  # tasks.jsonl is optional in some exports
            self.tasks = {}

        self.chunks_size = int(self.info.get("chunks_size", 1000))

    # ------------------------------------------------------------------
    # Asset resolution
    # ------------------------------------------------------------------

    def _fetch(self, rel_path: str) -> Path:
        """Return a local path for ``rel_path``, downloading it if necessary."""
        if self.local_path is not None:
            path = self.local_path / rel_path
            if not path.exists():
                raise FileNotFoundError(
                    f"{rel_path} not found under {self.local_path}. "
                    "Is this a LeRobot v2.1 dataset directory?"
                )
            return path

        hub = _require_hub()
        self._hf_cache.mkdir(parents=True, exist_ok=True)
        return Path(
            hub.hf_hub_download(
                repo_id=self.repo_id,
                filename=rel_path,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=str(self._hf_cache),
            )
        )

    def chunk_of(self, episode_index: int) -> int:
        """Return the chunk directory index holding ``episode_index``."""
        return int(episode_index) // self.chunks_size

    def parquet_path(self, episode_index: int) -> Path:
        """Local path to one episode's parquet, fetched on demand."""
        rel = self.info["data_path"].format(
            episode_chunk=self.chunk_of(episode_index),
            episode_index=int(episode_index),
        )
        return self._fetch(rel)

    def video_path(self, episode_index: int, video_key: str) -> Path:
        """Local path to one episode's mp4 for ``video_key``, fetched on demand."""
        rel = self.info["video_path"].format(
            episode_chunk=self.chunk_of(episode_index),
            video_key=video_key,
            episode_index=int(episode_index),
        )
        return self._fetch(rel)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Control frequency in steps per second.

        Read from the top-level ``fps`` field.  Note that some LeRobot exports
        carry a contradictory ``features[<video_key>].video_info["video.fps"]``
        -- that field describes the encoder and can disagree with the real
        frame rate; the top-level value and
        ``features[<video_key>].info["video.fps"]`` are the authoritative ones.
        """
        return float(self.info["fps"])

    @property
    def video_keys(self) -> List[str]:
        """Feature names whose dtype is ``video`` (one mp4 stream each)."""
        return [
            key
            for key, spec in self.info.get("features", {}).items()
            if spec.get("dtype") == "video"
        ]

    def get_episode_ids(self) -> List[int]:
        """Sorted episode indices present in this dataset."""
        return sorted(self.episodes_meta)

    def get_episode_length(self, episode_index: int) -> int:
        """Number of steps in one episode, per ``meta/episodes.jsonl``."""
        return int(self.episodes_meta[int(episode_index)]["length"])

    def task_for_episode(self, episode_index: int) -> Optional[str]:
        """Natural-language task for an episode, or ``None`` if unavailable.

        Prefers the ``tasks`` list carried in ``meta/episodes.jsonl``; falls
        back to the first entry of ``meta/tasks.jsonl``.
        """
        record = self.episodes_meta.get(int(episode_index), {})
        episode_tasks = record.get("tasks") or []
        if episode_tasks:
            return str(episode_tasks[0])
        if self.tasks:
            return str(self.tasks[min(self.tasks)])
        return None
