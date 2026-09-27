"""Offline tests for LeRobotVideoDataset against a synthetic LeRobot v2.1 tree.

Every test builds its own dataset directory in ``tmp_path`` -- real parquet
files and real (tiny) mp4s -- so the suite never touches the network or the
HuggingFace Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")
imageio = pytest.importorskip("imageio")
pytest.importorskip("pandas")

from robotdataset.lerobot.loader import LeRobotSource  # noqa: E402
from robotdataset.lerobot.memmap_builder import FrameCountMismatch  # noqa: E402
from robotdataset.lerobot_video_dataset import LeRobotVideoDataset  # noqa: E402

FPS = 10.0
HEIGHT = WIDTH = 32
CAMERAS = ("observation.images.top", "observation.images.wrist")


def _write_mp4(path: Path, n_frames: int, seed: int = 0) -> None:
    """Write a tiny mp4 with deterministic, visually distinct frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = [
        np.full((HEIGHT, WIDTH, 3), (seed * 37 + i * 11) % 256, dtype=np.uint8)
        + rng.integers(0, 2, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        for i in range(n_frames)
    ]
    imageio.mimwrite(str(path), frames, fps=FPS, macro_block_size=1)


def make_dataset(
    root: Path,
    episode_lengths=(6, 4),
    video_frames_override=None,
    with_aggregate_state: bool = True,
) -> Path:
    """Materialise a minimal but valid LeRobot v2.1 dataset directory.

    Args:
        root: Directory to create the tree in.
        episode_lengths: Number of steps per episode.
        video_frames_override: ``{episode_index: n_frames}`` forcing a video to
            disagree with its parquet, to exercise the mismatch guard.
        with_aggregate_state: Also emit the concatenated ``observation.state``
            column alongside its slices, which is the leaf/branch collision
            case.
    """
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)

    features = {
        cam: {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "names": ["height", "width", "channels"],
            # Deliberately contradictory: the encoder block claims 30 fps while
            # the real stream is FPS. Mirrors a real LeRobot export.
            "video_info": {"video.fps": 30.0, "video.codec": "h264"},
            "info": {"video.fps": FPS, "video.codec": "h264"},
        }
        for cam in CAMERAS
    }
    features["observation.state.joints"] = {"dtype": "float32", "shape": [3]}
    features["observation.state.gripper"] = {"dtype": "float32", "shape": [1]}
    if with_aggregate_state:
        features["observation.state"] = {"dtype": "float32", "shape": [4]}
    features["action"] = {"dtype": "float32", "shape": [2]}

    info = {
        "codebase_version": "v2.1",
        "fps": FPS,
        "total_episodes": len(episode_lengths),
        "total_frames": int(sum(episode_lengths)),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "stack the blocks."}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": ep,
                    "tasks": ["stack the blocks."],
                    "length": int(length),
                }
            )
            + "\n"
            for ep, length in enumerate(episode_lengths)
        )
    )

    global_index = 0
    for ep, length in enumerate(episode_lengths):
        rows = []
        for t in range(length):
            row = {
                "observation.state.joints": np.arange(3, dtype=np.float32) + t,
                "observation.state.gripper": np.array([float(t)], dtype=np.float32),
                "action": np.array([t * 0.1, -t * 0.1], dtype=np.float32),
                "timestamp": t / FPS,
                "frame_index": t,
                "episode_index": ep,
                "index": global_index,
                "task_index": 0,
            }
            if with_aggregate_state:
                row["observation.state"] = np.arange(4, dtype=np.float32) + t
            rows.append(row)
            global_index += 1
        pq_path = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(pq_path)

        n_frames = (video_frames_override or {}).get(ep, length)
        for cam_i, cam in enumerate(CAMERAS):
            _write_mp4(
                root / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4",
                n_frames,
                seed=ep * 10 + cam_i,
            )

    return root


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "lerobot_ds")


# ----------------------------------------------------------------------
# Source-level behaviour
# ----------------------------------------------------------------------


def test_source_parses_metadata(dataset_dir: Path, tmp_path: Path):
    source = LeRobotSource(local_path=str(dataset_dir), cache_dir=str(tmp_path / "c"))
    assert source.get_episode_ids() == [0, 1]
    assert source.get_episode_length(0) == 6
    assert source.get_episode_length(1) == 4
    assert source.fps == FPS
    assert sorted(source.video_keys) == sorted(CAMERAS)
    assert source.task_for_episode(0) == "stack the blocks."


def test_source_rejects_ambiguous_origin(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        LeRobotSource(repo_id="a/b", local_path=str(tmp_path))
    with pytest.raises(ValueError, match="exactly one"):
        LeRobotSource()


def test_source_resolves_asset_paths(dataset_dir: Path, tmp_path: Path):
    source = LeRobotSource(local_path=str(dataset_dir), cache_dir=str(tmp_path / "c"))
    assert source.parquet_path(1).name == "episode_000001.parquet"
    assert source.video_path(1, CAMERAS[0]).parent.name == CAMERAS[0]


# ----------------------------------------------------------------------
# Dataset-level behaviour
# ----------------------------------------------------------------------


def test_loads_only_selected_episodes(dataset_dir: Path, tmp_path: Path):
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir), episodes=[1], root=str(tmp_path / "cache"), batch_size=2
    )
    assert ds.num_episodes == 1
    assert len(ds) == 4  # episode 1 only

    cached = {p.name for p in ds.data_path.iterdir() if p.is_dir()}
    assert cached == {"1"}, "episode 0 should never have been decoded"


def test_ted_structure_and_episode_ids(dataset_dir: Path, tmp_path: Path):
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir), root=str(tmp_path / "cache"), batch_size=2
    )
    assert len(ds) == 10
    assert ds.num_episodes == 2

    paths = set(ds.get_modalities())
    assert "action" in paths
    assert "observation/images/top" in paths
    assert "observation/images/wrist" in paths
    assert "observation/state/joints" in paths
    assert "collector/episode_id" in paths
    assert any(p.startswith("next/observation/") for p in paths)

    kinds = {p: spec["kind"] for p, spec in ds.get_modalities().items()}
    assert kinds["observation/images/top"] == "image"
    assert kinds["observation/state/joints"] == "state"
    assert kinds["action"] == "action"

    episode_ids = ds._storage._storage["collector", "episode_id"]
    assert episode_ids[:6].unique().tolist() == [0]
    assert episode_ids[6:].unique().tolist() == [1]


def test_aggregate_state_column_is_demoted(dataset_dir: Path, tmp_path: Path):
    """observation.state and observation.state.joints must coexist."""
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir), root=str(tmp_path / "cache"), batch_size=2
    )
    modalities = ds.get_modalities()
    assert modalities["observation/state/all"]["shape"] == (4,)
    assert modalities["observation/state/joints"]["shape"] == (3,)


def test_no_aggregate_when_absent(tmp_path: Path):
    root = make_dataset(tmp_path / "ds2", with_aggregate_state=False)
    ds = LeRobotVideoDataset(
        local_path=str(root), root=str(tmp_path / "cache"), batch_size=2
    )
    assert "observation/state/all" not in ds.get_modalities()
    assert "observation/state/joints" in ds.get_modalities()


def test_images_delivered_channel_first(dataset_dir: Path, tmp_path: Path):
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir), root=str(tmp_path / "cache"), batch_size=3
    )
    assert ("observation", "images", "top") in ds.image_keys

    batch = ds.sample()
    # stored HWC on disk, permuted to (B, T, C, H, W) by the sampler
    assert batch["observation/images/top"].shape == (3, 1, 3, HEIGHT, WIDTH)
    assert ds.get_modalities()["observation/images/top"]["shape"] == (HEIGHT, WIDTH, 3)


def test_delta_timestamps_window(dataset_dir: Path, tmp_path: Path):
    deltas = [-0.2, -0.1, 0.0]  # 3 steps at 10 fps
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir),
        root=str(tmp_path / "cache"),
        batch_size=4,
        delta_timestamps={"observation/images/wrist": deltas},
    )
    batch = ds.sample()
    assert batch["observation/images/wrist"].shape == (4, 3, 3, HEIGHT, WIDTH)
    # unlisted modalities keep the default anchor-only window
    assert batch["action"].shape[:2] == (4, 1)


def test_control_frequency_defaults_to_dataset_fps(dataset_dir: Path, tmp_path: Path):
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir), root=str(tmp_path / "cache"), batch_size=2
    )
    assert ds.control_frequency == FPS
    assert ds.fps == FPS


def test_video_key_subset_skips_other_cameras(dataset_dir: Path, tmp_path: Path):
    ds = LeRobotVideoDataset(
        local_path=str(dataset_dir),
        root=str(tmp_path / "cache"),
        batch_size=2,
        video_keys=[CAMERAS[0]],
    )
    paths = set(ds.get_modalities())
    assert "observation/images/top" in paths
    assert "observation/images/wrist" not in paths


def test_second_construction_reuses_cache(dataset_dir: Path, tmp_path: Path):
    cache = str(tmp_path / "cache")
    first = LeRobotVideoDataset(
        local_path=str(dataset_dir), episodes=[0], root=cache, batch_size=2
    )
    assert first._is_downloaded()

    # Remove the source tree entirely: a cached rebuild must not need it.
    import shutil

    shutil.rmtree(dataset_dir / "videos")
    second = LeRobotVideoDataset(
        local_path=str(dataset_dir), episodes=[0], root=cache, batch_size=2
    )
    assert len(second) == len(first)


def test_unknown_episode_raises(dataset_dir: Path, tmp_path: Path):
    with pytest.raises(IndexError, match="not in this dataset"):
        LeRobotVideoDataset(
            local_path=str(dataset_dir), episodes=[99], root=str(tmp_path / "cache")
        )


def test_frame_count_mismatch_raises(tmp_path: Path):
    """A video shorter than its parquet must fail loudly, not silently truncate."""
    root = make_dataset(tmp_path / "ds_bad", video_frames_override={0: 3})
    with pytest.raises(FrameCountMismatch, match="frame rate"):
        LeRobotVideoDataset(
            local_path=str(root), episodes=[0], root=str(tmp_path / "cache")
        )
