from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("datasets", reason="hf extras not installed")

import robotdataset.table30v2_dataset as t30
from robotdataset.hf.loader import hf_episode_to_oxe_format, _convert_leaf, _to_nested, filter_by_tasks


# ---------------------------------------------------------------------------
# Fake HF dataset infrastructure
# ---------------------------------------------------------------------------

N_STEPS = 3
N_EPISODES = 5


def _make_row(episode_idx: int, frame_idx: int, n_steps: int = N_STEPS, task_idx: int = 0) -> dict:
    """Create one row of a fake LeRobot-style HF dataset."""
    return {
        "episode_index": episode_idx,
        "frame_index": frame_idx,
        "observation.image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.state": np.array([float(episode_idx)] * 4, dtype=np.float32),
        "action": np.array([float(episode_idx), float(frame_idx)], dtype=np.float32),
        "next.reward": np.float32(float(frame_idx)),
        "next.done": bool(frame_idx == n_steps - 1),
        "next.terminated": bool(frame_idx == n_steps - 1),
        "timestamp": float(frame_idx) * 0.1,
        "task_index": task_idx,
    }


def _make_all_rows(n_episodes: int = N_EPISODES, n_steps: int = N_STEPS) -> list:
    rows = []
    for ep in range(n_episodes):
        for f in range(n_steps):
            rows.append(_make_row(ep, f, n_steps))
    return rows


def _make_multitask_rows(n_episodes: int = N_EPISODES, n_steps: int = N_STEPS) -> list:
    """Rows where even episodes belong to task 0, odd episodes to task 1."""
    rows = []
    for ep in range(n_episodes):
        task = ep % 2  # ep 0,2,4 → task 0; ep 1,3 → task 1
        for f in range(n_steps):
            rows.append(_make_row(ep, f, n_steps, task_idx=task))
    return rows


_ALL_ROWS = _make_all_rows()


class FakeFeatures(dict):
    """Minimal stand-in for HF dataset.features."""
    pass


class FakeHFDataset:
    """Minimal HF Dataset stand-in for testing."""

    def __init__(self, rows: list):
        self._rows = list(rows)
        self.features = FakeFeatures({
            "episode_index": None,
            "frame_index": None,
            "observation.image": None,
            "observation.state": None,
            "action": None,
            "next.reward": None,
            "next.done": None,
            "next.terminated": None,
            "timestamp": None,
            "task_index": None,
        })

    def filter(self, fn):
        return FakeHFDataset([r for r in self._rows if fn(r)])

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[key]

    def __len__(self):
        return len(self._rows)


def _patch(monkeypatch: pytest.MonkeyPatch, rows=None):
    """Monkeypatch load_hf_dataset to return a FakeHFDataset."""
    data = rows if rows is not None else _ALL_ROWS
    fake = FakeHFDataset(data)
    monkeypatch.setattr(t30, "load_hf_dataset", lambda *a, **kw: fake)


# ---------------------------------------------------------------------------
# _convert_leaf
# ---------------------------------------------------------------------------

def test_convert_leaf_numpy_uint8() -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    out = _convert_leaf(arr)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.uint8
    assert out.shape == (8, 8, 3)


def test_convert_leaf_numpy_float32() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = _convert_leaf(arr)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32


def test_convert_leaf_numpy_scalar() -> None:
    out = _convert_leaf(np.float32(3.14))
    assert isinstance(out, torch.Tensor)
    assert out.item() == pytest.approx(3.14, abs=1e-5)


def test_convert_leaf_python_bool() -> None:
    out = _convert_leaf(True)
    assert isinstance(out, torch.Tensor)
    assert bool(out.item()) is True


def test_convert_leaf_python_list_of_floats() -> None:
    out = _convert_leaf([1.0, 2.0, 3.0])
    assert isinstance(out, torch.Tensor)
    assert out.tolist() == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# _to_nested
# ---------------------------------------------------------------------------

def test_to_nested_flat_keys() -> None:
    flat = {"a": 1, "b.c": 2, "b.d": 3}
    nested = _to_nested(flat)
    assert nested["a"] == 1
    assert nested["b"]["c"] == 2
    assert nested["b"]["d"] == 3


def test_to_nested_no_dots() -> None:
    flat = {"action": torch.zeros(2), "reward": torch.zeros(1)}
    nested = _to_nested(flat)
    assert set(nested.keys()) == {"action", "reward"}


# ---------------------------------------------------------------------------
# hf_episode_to_oxe_format
# ---------------------------------------------------------------------------

def test_hf_episode_to_oxe_format_structure() -> None:
    rows = [_make_row(0, t) for t in range(N_STEPS)]
    episode = hf_episode_to_oxe_format(rows)
    assert "steps" in episode
    assert len(episode["steps"]) == N_STEPS


def test_hf_episode_to_oxe_format_observation_nested() -> None:
    rows = [_make_row(0, t) for t in range(N_STEPS)]
    episode = hf_episode_to_oxe_format(rows)
    step = episode["steps"][0]
    assert "observation" in step
    assert "image" in step["observation"]
    assert "state" in step["observation"]


def test_hf_episode_to_oxe_format_next_lifted() -> None:
    rows = [_make_row(0, t) for t in range(N_STEPS)]
    episode = hf_episode_to_oxe_format(rows)
    step = episode["steps"][0]
    assert "is_last" in step
    assert "is_terminal" in step
    assert "reward" in step
    assert "next" not in step  # "next" sub-dict was lifted


def test_hf_episode_to_oxe_format_meta_removed() -> None:
    rows = [_make_row(0, t) for t in range(N_STEPS)]
    episode = hf_episode_to_oxe_format(rows)
    step = episode["steps"][0]
    assert "episode_index" not in step
    assert "frame_index" not in step
    assert "timestamp" not in step
    assert "task_index" not in step


def test_hf_episode_to_oxe_format_last_step_done() -> None:
    rows = [_make_row(0, t) for t in range(N_STEPS)]
    episode = hf_episode_to_oxe_format(rows)
    steps = episode["steps"]
    # last step is_last should be True
    last_is_last = steps[-1]["is_last"]
    first_is_last = steps[0]["is_last"]
    assert bool(last_is_last.item() if isinstance(last_is_last, torch.Tensor) else last_is_last)
    assert not bool(first_is_last.item() if isinstance(first_is_last, torch.Tensor) else first_is_last)


# ---------------------------------------------------------------------------
# Table30v2Dataset — construction and ReplayBuffer API
# ---------------------------------------------------------------------------

def test_dataset_has_correct_step_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """5 episodes × 3 steps = 15 total steps in the buffer."""
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", root=str(tmp_path))
    assert len(ds) == N_EPISODES * N_STEPS


def test_sample_returns_tensordict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", batch_size=4, root=str(tmp_path))
    batch = ds.sample()
    assert isinstance(batch, TensorDict)
    assert batch.batch_size[0] == 4


def test_sample_has_ted_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", batch_size=2, root=str(tmp_path))
    batch = ds.sample()
    assert "observation" in batch.keys()
    assert "action" in batch.keys()
    assert "done" in batch.keys()
    assert "next" in batch.keys()
    assert "observation" in batch["next"].keys()
    assert "reward" in batch["next"].keys()


def test_sample_action_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Action has T=1 by default — shape (B, T=1, action_dim)."""
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", batch_size=3, root=str(tmp_path))
    batch = ds.sample()
    assert isinstance(batch["action"], torch.Tensor)
    assert batch["action"].shape == (3, 1, 2)  # (B, T=1, action_dim=2)


def test_sample_image_channels_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Images are channels-first (B, T=1, C, H, W) by default."""
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", batch_size=2, root=str(tmp_path))
    batch = ds.sample()
    # Stored as HWC (8, 8, 3), sampled as CHW (3, 8, 8)
    assert batch["observation", "image"].shape == (2, 1, 3, 8, 8)
    assert batch["observation", "image"].dtype == torch.uint8


# ---------------------------------------------------------------------------
# Table30v2Dataset — episode filter
# ---------------------------------------------------------------------------

def test_num_episodes_property(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", root=str(tmp_path))
    assert ds.num_episodes == N_EPISODES


def test_episodes_filter_reduces_step_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """episodes=[0, 1] → 2 episodes × 3 steps = 6 steps."""
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0, 1], root=str(tmp_path))
    assert len(ds) == 6


def test_episodes_single(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[2], root=str(tmp_path))
    assert len(ds) == N_STEPS


def test_base_dataset_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from torchrl.data.datasets.common import BaseDatasetExperienceReplay
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", root=str(tmp_path))
    assert isinstance(ds, BaseDatasetExperienceReplay)
    assert ds.data_path == tmp_path / "hf" / "table30v2" / "train" / "episodes"
    assert ds.data_path_root == tmp_path / "hf" / "table30v2"
    assert ds._is_downloaded() is True


# ---------------------------------------------------------------------------
# Table30v2Dataset — per-episode caching
# ---------------------------------------------------------------------------

def test_per_episode_sentinels_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json as _json
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", root=str(tmp_path))
    episodes_dir = ds._episodes_dir()
    for i in range(N_EPISODES):
        sentinel = episodes_dir / str(i) / "_steps.json"
        assert sentinel.exists(), f"sentinel missing for episode {i}"
        meta = _json.loads(sentinel.read_text())
        assert meta["n_steps"] == N_STEPS


def test_no_rebuild_on_same_episodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Second init with same episodes calls build_missing_episodes 0 times."""
    _patch(monkeypatch)
    from robotdataset.hf import memmap_builder as mb

    calls = []
    original = mb.build_missing_episodes

    def spy(*a, **kw):
        calls.append(kw.get("missing", a[2] if len(a) > 2 else []))
        return original(*a, **kw)

    monkeypatch.setattr(t30, "build_missing_episodes", spy)

    t30.Table30v2Dataset(split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert sorted(calls[0]) == [0, 1, 2]

    calls.clear()
    t30.Table30v2Dataset(split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert calls == [], "build_missing_episodes must not be called when all episodes are cached"


def test_incremental_reuse_across_episode_lists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Second init with a superset only builds the new episodes."""
    _patch(monkeypatch)
    from robotdataset.hf import memmap_builder as mb

    built: list = []
    original = mb.build_missing_episodes

    def spy(*a, **kw):
        missing = kw.get("missing", a[2] if len(a) > 2 else [])
        built.extend(missing)
        return original(*a, **kw)

    monkeypatch.setattr(t30, "build_missing_episodes", spy)

    t30.Table30v2Dataset(split="train", episodes=[0, 2], root=str(tmp_path))
    assert sorted(built) == [0, 2]

    built.clear()
    t30.Table30v2Dataset(split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert built == [1], f"expected only episode 1 to be built, got {built}"


# ---------------------------------------------------------------------------
# Table30v2Dataset — combined memmap
# ---------------------------------------------------------------------------

def test_combined_storage_sentinel_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json as _json
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0, 1], root=str(tmp_path))
    sentinel = ds._combined_dir([0, 1]) / "_complete.json"
    assert sentinel.exists()
    meta = _json.loads(sentinel.read_text())
    assert meta["n_steps"] == 6
    assert meta["episodes"] == [0, 1]


def test_combined_storage_not_rebuilt_on_second_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    from robotdataset.oxe import memmap_builder as mb

    calls: list = []
    original = mb.build_combined_storage

    def spy(*a, **kw):
        calls.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(t30, "build_combined_storage", spy)

    t30.Table30v2Dataset(split="train", episodes=[0, 1], root=str(tmp_path))
    assert len(calls) == 1

    calls.clear()
    t30.Table30v2Dataset(split="train", episodes=[0, 1], root=str(tmp_path))
    assert calls == [], "build_combined_storage must not be called when already complete"


def test_combined_storage_is_memory_mapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tensordict.memmap import MemoryMappedTensor
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0, 1], root=str(tmp_path))
    storage_td = ds._storage._storage
    assert isinstance(storage_td["action"], MemoryMappedTensor)


# ---------------------------------------------------------------------------
# Table30v2Dataset — temporal sampling
# ---------------------------------------------------------------------------

def test_temporal_sampler_default_t1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default delta_timestamps → T=1 for all modalities."""
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0, 1, 2], batch_size=4, root=str(tmp_path))
    batch = ds.sample()
    assert batch["action"].shape == (4, 1, 2)
    assert batch["observation", "image"].shape == (4, 1, 3, 8, 8)


def test_temporal_sampler_multi_offset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """delta_timestamps with 3 offsets → T=3."""
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(
        split="train",
        episodes=[0, 1, 2],
        batch_size=4,
        root=str(tmp_path),
        delta_timestamps={"observation/image": [-0.1, 0.0, 0.1]},
        control_frequency=10.0,
    )
    batch = ds.sample()
    assert batch["observation", "image"].shape == (4, 3, 3, 8, 8)
    assert batch["observation", "image"].dtype == torch.uint8


def test_set_sampler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """set_sampler replaces the active sampler without rebuilding."""
    from robotdataset.oxe.temporal_sampler import TemporalSampler
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0, 1], batch_size=2, root=str(tmp_path))
    new_sampler = TemporalSampler(
        delta_timestamps={"action": [-0.1, 0.0]},
        control_frequency=10.0,
    )
    ds.set_sampler(new_sampler)
    batch = ds.sample()
    assert batch["action"].shape == (2, 2, 2)  # T=2 offsets


# ---------------------------------------------------------------------------
# Table30v2Dataset — modalities
# ---------------------------------------------------------------------------

def test_get_modalities_returns_image_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0], root=str(tmp_path))
    mods = ds.get_modalities()
    assert "observation/image" in mods
    assert mods["observation/image"]["kind"] == "image"


def test_image_keys_property(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", episodes=[0], root=str(tmp_path))
    assert ("observation", "image") in ds.image_keys


# ---------------------------------------------------------------------------
# filter_by_tasks — unit tests
# ---------------------------------------------------------------------------

def test_filter_by_tasks_reduces_rows() -> None:
    rows = _make_multitask_rows()
    ds = FakeHFDataset(rows)
    filtered = filter_by_tasks(ds, tasks=[0])
    # task 0 → episodes 0, 2, 4 → 3 episodes × 3 steps = 9 rows
    assert len(filtered) == 3 * N_STEPS


def test_filter_by_tasks_correct_episodes() -> None:
    rows = _make_multitask_rows()
    ds = FakeHFDataset(rows)
    filtered = filter_by_tasks(ds, tasks=[1])
    episode_ids = set(filtered["episode_index"])
    # task 1 → episodes 1, 3
    assert episode_ids == {1, 3}


def test_filter_by_tasks_multiple_tasks() -> None:
    rows = _make_multitask_rows()
    ds = FakeHFDataset(rows)
    filtered = filter_by_tasks(ds, tasks=[0, 1])
    assert len(filtered) == N_EPISODES * N_STEPS  # all episodes


def test_filter_by_tasks_no_task_column_raises() -> None:
    rows = [{"episode_index": 0, "frame_index": 0}]

    class NoTaskFeatures(dict):
        pass

    class DatasetNoTask:
        features = NoTaskFeatures({"episode_index": None, "frame_index": None})
        def __iter__(self): return iter(rows)
        def __getitem__(self, k): return [r[k] for r in rows]

    with pytest.raises(ValueError, match="task ID column"):
        filter_by_tasks(DatasetNoTask(), tasks=[0])


# ---------------------------------------------------------------------------
# Table30v2Dataset — tasks parameter
# ---------------------------------------------------------------------------

def _patch_multitask(monkeypatch: pytest.MonkeyPatch):
    fake = FakeHFDataset(_make_multitask_rows())
    monkeypatch.setattr(t30, "load_hf_dataset", lambda *a, **kw: fake)


def test_tasks_filter_selects_correct_episodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """tasks=[0] → episodes 0, 2, 4 → 3 × 3 steps = 9."""
    _patch_multitask(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", tasks=[0], root=str(tmp_path))
    assert len(ds) == 3 * N_STEPS
    assert ds.num_episodes == 3


def test_tasks_filter_odd_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """tasks=[1] → episodes 1, 3 → 2 × 3 steps = 6."""
    _patch_multitask(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", tasks=[1], root=str(tmp_path))
    assert len(ds) == 2 * N_STEPS
    assert ds.num_episodes == 2


def test_tasks_and_episodes_combined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """tasks=[0] then episodes=[0] → just episode 0, 3 steps."""
    _patch_multitask(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", tasks=[0], episodes=[0], root=str(tmp_path))
    assert len(ds) == N_STEPS
    assert ds.num_episodes == 1


def test_tasks_all_returns_all_episodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """tasks=None → all episodes."""
    _patch_multitask(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", tasks=None, root=str(tmp_path))
    assert ds.num_episodes == N_EPISODES


def test_tasks_stored_on_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_multitask(monkeypatch)
    ds = t30.Table30v2Dataset(split="train", tasks=[0], root=str(tmp_path))
    assert ds.tasks == [0]
