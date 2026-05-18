"""Tests for AgiBotWorldBetaDataset and agibot.* utilities.

All tests are offline — they patch `load_task_info`, `get_repo_files`,
`download_tar`, and `decode_mp4_from_tar` so no network or video-decoding
library is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import robotdataset.agibot_dataset as agibot_mod
from robotdataset.agibot.loader import (
    build_episode_mapping,
    find_tar_for_episode,
    load_episode_map,
    save_episode_map,
)


# ---------------------------------------------------------------------------
# Fake dataset fixtures
# ---------------------------------------------------------------------------

TASK_A = 100
TASK_B = 200

# Episodes per task — IDs intentionally non-sequential to mirror real data.
TASK_A_EPISODES: List[Dict] = [
    {"episode_id": 1000, "task_name": "Task A", "label_info": {"action_config": []}},
    {"episode_id": 1010, "task_name": "Task A", "label_info": {"action_config": []}},
    {"episode_id": 1020, "task_name": "Task A", "label_info": {"action_config": []}},
]
TASK_B_EPISODES: List[Dict] = [
    {"episode_id": 2000, "task_name": "Task B", "label_info": {"action_config": []}},
    {"episode_id": 2010, "task_name": "Task B", "label_info": {"action_config": []}},
]

FAKE_TASK_INFOS: Dict[int, List[Dict]] = {
    TASK_A: TASK_A_EPISODES,
    TASK_B: TASK_B_EPISODES,
}

# Repo files that satisfy find_tar_for_episode for all episodes above.
FAKE_REPO_FILES: List[str] = [
    f"task_info/task_{TASK_A}.json",
    f"task_info/task_{TASK_B}.json",
    f"observations/{TASK_A}/1000-1020.tar",
    f"observations/{TASK_B}/2000-2010.tar",
]

# Small fake video: 5 frames of 8×8 RGB.
N_FRAMES = 5
H, W = 8, 8
CAMERAS = ["head_color", "hand_left_color"]


def _fake_frames(*_args, **_kwargs) -> np.ndarray:
    """Stand-in for decode_mp4_from_tar — returns tiny zero frames."""
    return np.zeros((N_FRAMES, H, W, 3), dtype=np.uint8)


def _fake_tar_path(*_args, **_kwargs) -> Path:
    """Stand-in for download_tar — returns an arbitrary Path (never opened)."""
    return Path("/fake/tar")


def _patch_all(monkeypatch: pytest.MonkeyPatch, task_infos=None):
    """Apply all required patches for offline testing."""
    infos = task_infos if task_infos is not None else FAKE_TASK_INFOS
    monkeypatch.setattr(agibot_mod, "load_task_info", lambda tid, *a, **kw: infos[tid])
    monkeypatch.setattr(agibot_mod, "get_repo_files", lambda: FAKE_REPO_FILES)
    monkeypatch.setattr(
        "robotdataset.agibot.memmap_builder.download_tar", _fake_tar_path
    )
    monkeypatch.setattr(
        "robotdataset.agibot.memmap_builder.decode_mp4_from_tar", _fake_frames
    )


# ---------------------------------------------------------------------------
# build_episode_mapping — unit tests
# ---------------------------------------------------------------------------

def test_build_episode_mapping_total_count() -> None:
    mapping = build_episode_mapping(FAKE_TASK_INFOS)
    assert len(mapping) == 3 + 2  # TASK_A + TASK_B


def test_build_episode_mapping_global_ids_unique() -> None:
    mapping = build_episode_mapping(FAKE_TASK_INFOS)
    global_ids = list(mapping.keys())
    assert sorted(global_ids) == list(range(len(global_ids)))


def test_build_episode_mapping_sorted_task_order() -> None:
    """Lower task_id (TASK_A=100) gets lower global IDs than TASK_B=200."""
    mapping = build_episode_mapping(FAKE_TASK_INFOS)
    a_ids = [gid for gid, (tid, _) in mapping.items() if tid == TASK_A]
    b_ids = [gid for gid, (tid, _) in mapping.items() if tid == TASK_B]
    assert max(a_ids) < min(b_ids)


def test_build_episode_mapping_episode_ids_preserved() -> None:
    mapping = build_episode_mapping(FAKE_TASK_INFOS)
    a_ep_ids = sorted(eid for _, (tid, eid) in mapping.items() if tid == TASK_A)
    assert a_ep_ids == [1000, 1010, 1020]


def test_build_episode_mapping_single_task() -> None:
    mapping = build_episode_mapping({TASK_A: TASK_A_EPISODES})
    assert len(mapping) == 3
    assert all(tid == TASK_A for _, (tid, _) in mapping.items())


# ---------------------------------------------------------------------------
# save_episode_map / load_episode_map — unit tests
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path: Path) -> None:
    original = {0: (100, 1000), 1: (100, 1010), 2: (200, 2000)}
    path = tmp_path / "episode_map.json"
    save_episode_map(path, original)
    restored = load_episode_map(path)
    assert restored == original


def test_load_episode_map_integer_types(tmp_path: Path) -> None:
    path = tmp_path / "episode_map.json"
    save_episode_map(path, {5: (327, 648649)})
    restored = load_episode_map(path)
    k, (tid, eid) = list(restored.items())[0]
    assert isinstance(k, int)
    assert isinstance(tid, int)
    assert isinstance(eid, int)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "episode_map.json"
    save_episode_map(path, {0: (1, 2)})
    assert path.exists()


# ---------------------------------------------------------------------------
# find_tar_for_episode — unit tests
# ---------------------------------------------------------------------------

def test_find_tar_task_a() -> None:
    result = find_tar_for_episode(TASK_A, 1000, FAKE_REPO_FILES)
    assert result == f"observations/{TASK_A}/1000-1020.tar"


def test_find_tar_task_a_middle_episode() -> None:
    result = find_tar_for_episode(TASK_A, 1010, FAKE_REPO_FILES)
    assert result == f"observations/{TASK_A}/1000-1020.tar"


def test_find_tar_task_b() -> None:
    result = find_tar_for_episode(TASK_B, 2010, FAKE_REPO_FILES)
    assert result == f"observations/{TASK_B}/2000-2010.tar"


def test_find_tar_missing_raises() -> None:
    with pytest.raises(ValueError, match="No TAR found"):
        find_tar_for_episode(TASK_A, 9999, FAKE_REPO_FILES)


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — construction and step count
# ---------------------------------------------------------------------------

def test_step_count_single_task(monkeypatch, tmp_path):
    """3 episodes × 5 frames = 15 steps."""
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, root=str(tmp_path)
    )
    assert len(ds) == 3 * N_FRAMES


def test_step_count_two_tasks(monkeypatch, tmp_path):
    """(3 + 2) episodes × 5 frames = 25 steps."""
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A, TASK_B], cameras=CAMERAS, root=str(tmp_path)
    )
    assert len(ds) == 5 * N_FRAMES


def test_num_episodes_single_task(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, root=str(tmp_path)
    )
    assert ds.num_episodes == 3


def test_num_episodes_two_tasks(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A, TASK_B], cameras=CAMERAS, root=str(tmp_path)
    )
    assert ds.num_episodes == 5


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — episode filter
# ---------------------------------------------------------------------------

def test_episodes_filter_reduces_count(monkeypatch, tmp_path):
    """episodes=[0, 1] → 2 × 5 = 10 steps."""
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A, TASK_B], cameras=CAMERAS, episodes=[0, 1], root=str(tmp_path)
    )
    assert len(ds) == 2 * N_FRAMES
    assert ds.num_episodes == 2


def test_episodes_filter_single(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0], root=str(tmp_path)
    )
    assert len(ds) == N_FRAMES
    assert ds.num_episodes == 1


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — TED format and sampling
# ---------------------------------------------------------------------------

def test_sample_returns_tensordict(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, batch_size=4, root=str(tmp_path)
    )
    batch = ds.sample()
    assert isinstance(batch, TensorDict)
    assert batch.batch_size[0] == 4


def test_sample_has_ted_keys(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, batch_size=2, root=str(tmp_path)
    )
    batch = ds.sample()
    assert "observation" in batch.keys()
    assert "action" in batch.keys()
    assert "done" in batch.keys()
    assert "next" in batch.keys()
    assert "observation" in batch["next"].keys()
    assert "reward" in batch["next"].keys()


def test_sample_camera_shape_channels_first(monkeypatch, tmp_path):
    """head_color stored as HWC (8,8,3); sampled as (B, T=1, C, H, W)."""
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, batch_size=3, root=str(tmp_path)
    )
    batch = ds.sample()
    img = batch["observation", "head_color"]
    assert img.shape == (3, 1, 3, H, W)
    assert img.dtype == torch.uint8


def test_sample_all_cameras_present(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, batch_size=2, root=str(tmp_path)
    )
    batch = ds.sample()
    for cam in CAMERAS:
        assert cam in batch["observation"].keys(), f"Camera {cam!r} missing from batch"


def test_action_is_zero_placeholder(monkeypatch, tmp_path):
    """Action vector is a zero placeholder (no ground-truth actions in AgiBot)."""
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, batch_size=4, root=str(tmp_path)
    )
    batch = ds.sample()
    assert batch["action"].shape == (4, 1, 1)  # (B, T=1, action_dim=1)
    assert (batch["action"] == 0).all()


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — temporal sampling
# ---------------------------------------------------------------------------

def test_temporal_sampler_multi_offset(monkeypatch, tmp_path):
    """3 time offsets → (B, T=3, C, H, W) camera tensor."""
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A],
        cameras=CAMERAS,
        episodes=[0, 1, 2],
        batch_size=4,
        root=str(tmp_path),
        delta_timestamps={"observation/head_color": [-1 / 30, 0.0, 1 / 30]},
        control_frequency=30.0,
    )
    batch = ds.sample()
    assert batch["observation", "head_color"].shape == (4, 3, 3, H, W)


def test_set_sampler(monkeypatch, tmp_path):
    """set_sampler replaces the active sampler without rebuilding."""
    from robotdataset.oxe.temporal_sampler import TemporalSampler

    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 1], batch_size=2, root=str(tmp_path)
    )
    new_sampler = TemporalSampler(
        delta_timestamps={"observation/head_color": [0.0, 1 / 30]},
        control_frequency=30.0,
        image_keys=ds.image_keys,
    )
    ds.set_sampler(new_sampler)
    batch = ds.sample()
    assert batch["observation", "head_color"].shape == (2, 2, 3, H, W)


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — modalities
# ---------------------------------------------------------------------------

def test_get_modalities_has_cameras(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0], root=str(tmp_path)
    )
    mods = ds.get_modalities()
    for cam in CAMERAS:
        key = f"observation/{cam}"
        assert key in mods, f"{key!r} not in modalities"
        assert mods[key]["kind"] == "image"


def test_image_keys_property(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0], root=str(tmp_path)
    )
    assert ("observation", "head_color") in ds.image_keys


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — BaseDatasetExperienceReplay interface
# ---------------------------------------------------------------------------

def test_data_path_properties(monkeypatch, tmp_path):
    from torchrl.data.datasets.common import BaseDatasetExperienceReplay

    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, root=str(tmp_path)
    )
    assert isinstance(ds, BaseDatasetExperienceReplay)
    assert ds.data_path == tmp_path / "hf" / "agibotworld-beta" / "train" / "episodes"
    assert ds.data_path_root == tmp_path / "hf" / "agibotworld-beta"
    assert ds._is_downloaded() is True


# ---------------------------------------------------------------------------
# AgiBotWorldBetaDataset — caching
# ---------------------------------------------------------------------------

def test_episode_map_persisted(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, root=str(tmp_path)
    )
    map_path = (
        tmp_path / "hf" / "agibotworld-beta" / "train" / "episodes" / "episode_map.json"
    )
    assert map_path.exists()
    data = json.loads(map_path.read_text())
    assert len(data) == 3  # TASK_A has 3 episodes


def test_per_episode_sentinels_created(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, root=str(tmp_path)
    )
    for i in range(3):
        sentinel = ds._episodes_dir() / str(i) / "_steps.json"
        assert sentinel.exists(), f"sentinel missing for episode {i}"
        meta = json.loads(sentinel.read_text())
        assert meta["n_steps"] == N_FRAMES


def test_no_rebuild_on_second_init(monkeypatch, tmp_path):
    """build_missing_agibot_episodes is not called when all episodes are cached."""
    _patch_all(monkeypatch)

    calls: list = []
    original = agibot_mod.build_missing_agibot_episodes

    def spy(*a, **kw):
        calls.extend(kw.get("missing", a[3] if len(a) > 3 else []))
        return original(*a, **kw)

    monkeypatch.setattr(agibot_mod, "build_missing_agibot_episodes", spy)

    agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 1], root=str(tmp_path)
    )
    assert sorted(calls) == [0, 1]

    calls.clear()
    agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 1], root=str(tmp_path)
    )
    assert calls == [], "build_missing_agibot_episodes must not fire when all cached"


def test_incremental_reuse(monkeypatch, tmp_path):
    """Second init with a superset only builds the new episode."""
    _patch_all(monkeypatch)

    built: list = []
    original = agibot_mod.build_missing_agibot_episodes

    def spy(*a, **kw):
        built.extend(kw.get("missing", a[3] if len(a) > 3 else []))
        return original(*a, **kw)

    monkeypatch.setattr(agibot_mod, "build_missing_agibot_episodes", spy)

    agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 2], root=str(tmp_path)
    )
    assert sorted(built) == [0, 2]

    built.clear()
    agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 1, 2], root=str(tmp_path)
    )
    assert built == [1], f"expected only episode 1 to be built; got {built}"


def test_combined_sentinel(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 1], root=str(tmp_path)
    )
    sentinel = ds._combined_dir([0, 1]) / "_complete.json"
    assert sentinel.exists()
    meta = json.loads(sentinel.read_text())
    assert meta["n_steps"] == 2 * N_FRAMES


def test_combined_is_memory_mapped(monkeypatch, tmp_path):
    from tensordict.memmap import MemoryMappedTensor

    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A], cameras=CAMERAS, episodes=[0, 1], root=str(tmp_path)
    )
    assert isinstance(ds._storage._storage["action"], MemoryMappedTensor)


# ---------------------------------------------------------------------------
# Multi-task episode mapping isolation
# ---------------------------------------------------------------------------

def test_multi_task_global_ids_contiguous(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A, TASK_B], cameras=CAMERAS, root=str(tmp_path)
    )
    global_ids = sorted(ds.episode_map.keys())
    assert global_ids == list(range(5))


def test_multi_task_episode_pairs_unique(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A, TASK_B], cameras=CAMERAS, root=str(tmp_path)
    )
    pairs = list(ds.episode_map.values())
    assert len(pairs) == len(set(pairs))


def test_episode_map_contains_correct_task_ids(monkeypatch, tmp_path):
    _patch_all(monkeypatch)
    ds = agibot_mod.AgiBotWorldBetaDataset(
        tasks=[TASK_A, TASK_B], cameras=CAMERAS, root=str(tmp_path)
    )
    task_ids_in_map = {tid for _, (tid, _) in ds.episode_map.items()}
    assert task_ids_in_map == {TASK_A, TASK_B}
