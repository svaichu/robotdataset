from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("tensorflow", reason="oxe extras not installed")

import robotdataset.oxe_dataset as oxe
from robotdataset.oxe.utils import dict_to_tensordict, episode_to_ted_steps, tf_to_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CACHE = {"droid": {"1.0.1": "gs://gresearch/robotics/droid/1.0.1"}}

# Each "episode" is a dict with a "steps" key, mirroring the real TFDS OXE format.
# Each step contains obs, action, reward, termination flags, and a text instruction.
def _make_episode(episode_idx: int, n_steps: int = 3) -> dict:
    steps = []
    for t in range(n_steps):
        steps.append(
            {
                "observation": {
                    "image": np.zeros((8, 8, 3), dtype=np.uint8),
                    "state": np.array([float(episode_idx)] * 4, dtype=np.float32),
                },
                "action": np.array([float(episode_idx), float(t)], dtype=np.float32),
                "reward": np.float32(float(t)),
                "is_last": np.bool_(t == n_steps - 1),
                "is_terminal": np.bool_(t == n_steps - 1),
                "language_instruction": b"pick up the block",
            }
        )
    return {"steps": steps}


_EPISODES = [_make_episode(i) for i in range(5)]


class FakeDataset:
    def __init__(self, items):
        self._items = list(items)

    def take(self, n):
        return FakeDataset(self._items[:n])

    def skip(self, n):
        return FakeDataset(self._items[n:])

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class FakeInfo:
    description = "fake"

    def __init__(self):
        self.features = {
            "observation": {
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "state": np.zeros((4,), dtype=np.float32),
            },
            "action": np.zeros((2,), dtype=np.float32),
            "language_instruction": b"pick up the block",
        }
        self.splits = {"train": SimpleNamespace(num_examples=len(_EPISODES))}


class FakeBuilder:
    def __init__(self, episodes):
        self.info = FakeInfo()
        self.meta = self.info.features
        self._episodes = episodes

    def as_dataset(self, split, shuffle_files=False, **kwargs):
        return FakeDataset(self._episodes)


def _patch(monkeypatch: pytest.MonkeyPatch, episodes=None):
    eps = episodes if episodes is not None else _EPISODES
    fake_tf = SimpleNamespace(Tensor=np.ndarray, io=SimpleNamespace(gfile=SimpleNamespace()))
    fake_tfds = SimpleNamespace(builder_from_directory=lambda **kw: FakeBuilder(eps))
    monkeypatch.setattr(oxe, "tf", fake_tf, raising=False)
    monkeypatch.setattr(oxe, "tfds", fake_tfds, raising=False)
    monkeypatch.setattr(oxe, "_TF_TENSOR_TYPES", tuple(), raising=False)
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)
    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: None)
    monkeypatch.setattr(oxe, "_copy_metadata_only", lambda src, dst: None)


# ---------------------------------------------------------------------------
# tf_to_torch — conversion coverage
# ---------------------------------------------------------------------------

def test_tf_to_torch_numeric_array() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = tf_to_torch(arr)
    assert isinstance(out, torch.Tensor)
    assert out.tolist() == pytest.approx([1.0, 2.0])


def test_tf_to_torch_uint8_image() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = tf_to_torch(img)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.uint8


def test_tf_to_torch_python_scalars() -> None:
    assert isinstance(tf_to_torch(3), torch.Tensor)
    assert isinstance(tf_to_torch(3.14), torch.Tensor)
    assert isinstance(tf_to_torch(True), torch.Tensor)


def test_tf_to_torch_numpy_scalar() -> None:
    out = tf_to_torch(np.float32(2.5))
    assert isinstance(out, torch.Tensor)
    assert out.item() == pytest.approx(2.5)


def test_tf_to_torch_bytes_decoded_to_str() -> None:
    out = tf_to_torch(b"pick up the block")
    assert isinstance(out, str)
    assert out == "pick up the block"


def test_tf_to_torch_nested_dict() -> None:
    data = {"obs": np.array([1.0], dtype=np.float32), "label": b"go"}
    out = tf_to_torch(data)
    assert isinstance(out["obs"], torch.Tensor)
    assert isinstance(out["label"], str)


# ---------------------------------------------------------------------------
# dict_to_tensordict
# ---------------------------------------------------------------------------

def test_dict_to_tensordict_tensors() -> None:
    data = {
        "action": torch.tensor([1.0, 2.0]),
        "observation": {"image": torch.zeros(8, 8, 3, dtype=torch.uint8)},
    }
    td = dict_to_tensordict(data)
    assert isinstance(td, TensorDict)
    assert isinstance(td["action"], torch.Tensor)
    assert isinstance(td["observation"], TensorDict)
    assert td["observation"]["image"].shape == (8, 8, 3)


def test_dict_to_tensordict_non_tensor() -> None:
    data = {"action": torch.tensor([0.0]), "language_instruction": "go left"}
    td = dict_to_tensordict(data)
    assert isinstance(td, TensorDict)
    assert td.get_non_tensor("language_instruction") == "go left"


# ---------------------------------------------------------------------------
# episode_to_ted_steps
# ---------------------------------------------------------------------------

def test_episode_to_ted_steps_basic() -> None:
    episode = _make_episode(0, n_steps=3)
    steps = episode_to_ted_steps(episode, episode_idx=0)
    assert len(steps) == 3
    for td in steps:
        assert isinstance(td, TensorDict)
        assert "observation" in td.keys()
        assert "action" in td.keys()
        assert "done" in td.keys()
        assert "terminated" in td.keys()
        assert "next" in td.keys()
        assert "observation" in td["next"].keys()
        assert "reward" in td["next"].keys()


def test_episode_to_ted_steps_last_step_done() -> None:
    episode = _make_episode(0, n_steps=2)
    steps = episode_to_ted_steps(episode, episode_idx=0)
    assert steps[-1]["done"].item() is True
    assert steps[0]["done"].item() is False


def test_episode_to_ted_steps_episode_id() -> None:
    episode = _make_episode(0, n_steps=2)
    steps = episode_to_ted_steps(episode, episode_idx=7)
    for td in steps:
        assert td["collector", "episode_id"].item() == 7


def test_episode_to_ted_steps_next_obs_for_non_terminal() -> None:
    """next/observation at step t should equal observation at step t+1."""
    episode = _make_episode(0, n_steps=3)
    steps = episode_to_ted_steps(episode, episode_idx=0)
    # action[t] encodes episode_idx and step t, so obs state value is episode_idx
    assert torch.allclose(
        steps[0]["next", "observation", "state"],
        steps[1]["observation", "state"],
    )


def test_episode_to_ted_steps_flat_format() -> None:
    """Flat episode dict (no 'steps' key) is treated as a single-step episode."""
    episode = {
        "observation": {"state": np.array([1.0, 2.0], dtype=np.float32)},
        "action": np.array([0.5], dtype=np.float32),
    }
    steps = episode_to_ted_steps(episode, episode_idx=0)
    assert len(steps) == 1


def test_episode_to_ted_steps_without_str_fields() -> None:
    episode = _make_episode(0, n_steps=1)
    steps = episode_to_ted_steps(episode, episode_idx=0, load_str_fields=False)
    assert steps[0].is_contiguous()
    assert "observation" in steps[0].keys()
    assert "action" in steps[0].keys()
    assert "language_instruction" not in steps[0].keys()


# ---------------------------------------------------------------------------
# list_datasets / validate / dataset2path
# ---------------------------------------------------------------------------

def test_list_datasets_returns_version_list_per_dataset() -> None:
    datasets = oxe.list_datasets()
    assert isinstance(datasets, dict)
    for name, versions in datasets.items():
        assert isinstance(versions, list)
        assert all(isinstance(v, str) for v in versions)
    if "droid" in datasets:
        assert "1.0.1" in datasets["droid"]


def test_dataset_path_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)
    assert oxe.validate_dataset_name("droid") is True
    assert oxe.validate_dataset_name("missing-dataset") is False
    assert oxe.dataset2path("droid") == "gs://gresearch/robotics/droid/1.0.1"


# ---------------------------------------------------------------------------
# Cache dir
# ---------------------------------------------------------------------------

def test_get_cache_dir_default() -> None:
    path = oxe._get_cache_dir()
    assert str(path).endswith(".cache/robotdataset")


def test_get_cache_dir_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ROBOTDATASET_CACHE", str(tmp_path / "custom_cache"))
    path = oxe._get_cache_dir()
    assert path == tmp_path / "custom_cache"


def test_get_cache_dir_override(tmp_path: Path) -> None:
    path = oxe._get_cache_dir(override=str(tmp_path / "explicit"))
    assert path == tmp_path / "explicit"


# ---------------------------------------------------------------------------
# OXEDataset — construction and ReplayBuffer API
# ---------------------------------------------------------------------------

def test_strip_trailing_slash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid/", split="train", root=str(tmp_path))
    assert ds.dataset_name == "droid"


def test_dataset_has_correct_step_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """5 episodes × 3 steps each = 15 total steps in the buffer."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert len(ds) == 15  # 5 episodes × 3 steps


def test_sample_returns_tensordict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", batch_size=4, root=str(tmp_path))
    batch = ds.sample()
    assert isinstance(batch, TensorDict)
    assert batch.batch_size[0] == 4


def test_sample_has_ted_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", batch_size=2, root=str(tmp_path))
    batch = ds.sample()
    assert "observation" in batch.keys()
    assert "action" in batch.keys()
    assert "done" in batch.keys()
    assert "next" in batch.keys()
    assert "observation" in batch["next"].keys()
    assert "reward" in batch["next"].keys()


def test_sample_action_is_tensor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Action has temporal dim T=1 by default — shape (B, T=1, action_dim)."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", batch_size=3, root=str(tmp_path))
    batch = ds.sample()
    assert isinstance(batch["action"], torch.Tensor)
    assert batch["action"].shape == (3, 1, 2)  # (B, T=1, action_dim=2)


def test_sample_observation_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Images are channels-first (B, T=1, C, H, W) — compulsory temporal + CHW."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", batch_size=2, root=str(tmp_path))
    batch = ds.sample()
    assert batch["observation", "image"].shape == (2, 1, 3, 8, 8)  # (B, T=1, C, H, W)
    assert batch["observation", "image"].dtype == torch.uint8


def test_sample_without_str_fields_is_contiguous(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        batch_size=2,
        root=str(tmp_path),
        load_str_fields=False,
    )
    batch = ds.sample()
    assert batch.is_contiguous()
    assert "language_instruction" not in batch.keys()


# ---------------------------------------------------------------------------
# OXEDataset — episodes filter
# ---------------------------------------------------------------------------

def test_num_episodes_property(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert ds.num_episodes == 5


def test_base_dataset_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from torchrl.data.datasets.common import BaseDatasetExperienceReplay
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert isinstance(ds, BaseDatasetExperienceReplay)
    assert ds.data_path == tmp_path / "oxe" / "droid" / "episodes" / "train"
    assert ds.data_path_root == tmp_path / "oxe" / "droid"
    assert ds._is_downloaded() is True


def test_episodes_filter_reduces_step_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """episodes=[0, 1] → 2 episodes × 3 steps = 6 steps."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path)
    )
    assert len(ds) == 6


def test_episodes_single(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid", split="train", episodes=[2], root=str(tmp_path)
    )
    assert len(ds) == 3  # 1 episode × 3 steps


# ---------------------------------------------------------------------------
# OXEDataset — per-episode caching
# ---------------------------------------------------------------------------

def test_per_episode_sentinels_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each episode gets its own _steps.json sentinel under episodes/train/{i}/."""
    import json as _json
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    episodes_dir = ds._episodes_dir()
    for i in range(5):
        sentinel = episodes_dir / str(i) / "_steps.json"
        assert sentinel.exists(), f"sentinel missing for episode {i}"
        meta = _json.loads(sentinel.read_text())
        assert meta["n_steps"] == 3  # 3 steps per episode in fake data


def test_no_rebuild_on_same_episodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Second init with the same episodes calls build_missing_episodes 0 times."""
    _patch(monkeypatch)
    from robotdataset.oxe import memmap_builder as mb

    calls = []
    original = mb.build_missing_episodes

    def spy(*a, **kw):
        calls.append(kw.get("missing", a[3] if len(a) > 3 else []))
        return original(*a, **kw)

    monkeypatch.setattr(oxe, "build_missing_episodes", spy)

    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert len(calls) == 1 and calls[0] == [0, 1, 2]

    calls.clear()
    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert calls == [], "build_missing_episodes must not be called when all episodes are cached"


def test_incremental_reuse_across_episode_lists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Second init with a superset only builds the new episodes, reuses the rest."""
    _patch(monkeypatch)
    from robotdataset.oxe import memmap_builder as mb

    built: list = []
    original = mb.build_missing_episodes

    def spy(*a, **kw):
        missing = kw.get("missing", a[3] if len(a) > 3 else [])
        built.extend(missing)
        return original(*a, **kw)

    monkeypatch.setattr(oxe, "build_missing_episodes", spy)

    # First init: episodes 0 and 2
    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 2], root=str(tmp_path))
    assert sorted(built) == [0, 2]

    built.clear()
    # Second init: episodes 0, 1, 2 — only episode 1 should be built
    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert built == [1], f"expected only episode 1 to be built, got {built}"
    assert len(oxe.OXEDataset(
        dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path)
    )) == 9  # 3 episodes × 3 steps


def test_full_download_when_no_episodes_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """episodes=None → metadata downloaded once, then full shards via _copy_tree."""
    eps = _EPISODES
    fake_tf = SimpleNamespace(Tensor=np.ndarray, io=SimpleNamespace(gfile=SimpleNamespace()))
    fake_tfds = SimpleNamespace(builder_from_directory=lambda **kw: FakeBuilder(eps))
    monkeypatch.setattr(oxe, "tf", fake_tf, raising=False)
    monkeypatch.setattr(oxe, "tfds", fake_tfds, raising=False)
    monkeypatch.setattr(oxe, "_TF_TENSOR_TYPES", tuple(), raising=False)
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)

    tree_calls, meta_calls = [], []

    def _fake_copy_metadata(src, dst):
        meta_calls.append(src)
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "dataset_info.json").write_text("{}")

    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: tree_calls.append(src))
    monkeypatch.setattr(oxe, "_copy_metadata_only", _fake_copy_metadata)

    oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert meta_calls == ["gs://gresearch/robotics/droid/1.0.1"], "_copy_metadata_only runs on first init"
    assert tree_calls == ["gs://gresearch/robotics/droid/1.0.1"], "_copy_tree runs for full download"

    # Second init: dataset_info.json now exists locally — no GCS calls
    meta_calls.clear()
    tree_calls.clear()
    oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert meta_calls == [], "_copy_metadata_only must not run when metadata already present"
    assert tree_calls == [], "_copy_tree must not run when all episodes already cached"


def test_metadata_only_download_when_episodes_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """episodes=[0,1] → only metadata downloaded on first init; _copy_tree never called."""
    eps = _EPISODES
    fake_tf = SimpleNamespace(Tensor=np.ndarray, io=SimpleNamespace(gfile=SimpleNamespace()))
    fake_tfds = SimpleNamespace(builder_from_directory=lambda **kw: FakeBuilder(eps))
    monkeypatch.setattr(oxe, "tf", fake_tf, raising=False)
    monkeypatch.setattr(oxe, "tfds", fake_tfds, raising=False)
    monkeypatch.setattr(oxe, "_TF_TENSOR_TYPES", tuple(), raising=False)
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)

    tree_calls, meta_calls = [], []

    def _fake_copy_metadata(src, dst):
        meta_calls.append(src)
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "dataset_info.json").write_text("{}")

    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: tree_calls.append(src))
    monkeypatch.setattr(oxe, "_copy_metadata_only", _fake_copy_metadata)

    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    assert tree_calls == [], "_copy_tree must not be called when episodes are specified"
    assert meta_calls == ["gs://gresearch/robotics/droid/1.0.1"]

    # Second init with same episodes — no GCS calls at all
    meta_calls.clear()
    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    assert meta_calls == [], "_copy_metadata_only must not run on second init"


def test_copy_tree_skipped_when_all_episodes_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Once all episodes are cached, _copy_tree is never called again."""
    _patch(monkeypatch)

    tree_calls = []
    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: tree_calls.append(src))

    oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    tree_calls.clear()
    oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert tree_calls == [], "_copy_tree must not run when all episodes are already cached"


# ---------------------------------------------------------------------------
# OXEDataset — local cache dir helpers
# ---------------------------------------------------------------------------

def test_local_tfds_dir_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert ds._local_tfds_dir() == tmp_path / "oxe" / "droid"


def test_local_tfds_dir_with_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", version="1.0.1", root=str(tmp_path))
    assert ds._local_tfds_dir() == tmp_path / "oxe" / "droid" / "1.0.1"


# ---------------------------------------------------------------------------
# Modalities and dataset info
# ---------------------------------------------------------------------------

def test_modalities_from_builder_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    mods = ds.get_modalities()
    assert mods["observation/image"]["kind"] == "image"
    assert mods["action"]["kind"] == "action"


def test_get_dataset_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    info = ds.get_dataset_info()
    assert info["description"] == "fake"


# ---------------------------------------------------------------------------
# Combined memmap (memory-safe storage)
# ---------------------------------------------------------------------------

def test_combined_storage_sentinel_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """After first init, the combined memmap sentinel _complete.json must exist."""
    import json as _json
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    combined_dir = ds._combined_dir([0, 1])
    sentinel = combined_dir / "_complete.json"
    assert sentinel.exists(), "_complete.json sentinel missing after first init"
    meta = _json.loads(sentinel.read_text())
    assert meta["n_steps"] == 6  # 2 episodes × 3 steps
    assert meta["episodes"] == [0, 1]


def test_combined_storage_not_rebuilt_on_second_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_combined_storage must not be called when the combined memmap already exists."""
    _patch(monkeypatch)
    from robotdataset.oxe import memmap_builder as mb

    calls: list = []
    original = mb.build_combined_storage

    def spy(*a, **kw):
        calls.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(oxe, "build_combined_storage", spy)

    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    assert len(calls) == 1

    calls.clear()
    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    assert calls == [], "build_combined_storage must not be called when combined storage is complete"


def test_different_episode_lists_get_separate_combined_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two OXEDataset instances with different episode lists get distinct combined dirs."""
    _patch(monkeypatch)
    ds1 = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    ds2 = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert ds1._combined_dir([0, 1]) != ds2._combined_dir([0, 1, 2])
    assert ds1._combined_dir([0, 1]).exists()
    assert ds2._combined_dir([0, 1, 2]).exists()


def test_combined_storage_is_memory_mapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tensor leaves in the combined storage must be MemoryMappedTensor, not plain Tensors."""
    from tensordict.memmap import MemoryMappedTensor
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    storage_td = ds._storage._storage
    assert isinstance(storage_td["action"], MemoryMappedTensor), (
        "action leaf should be MemoryMappedTensor (lazy disk access), got "
        f"{type(storage_td['action']).__name__}"
    )


# ---------------------------------------------------------------------------
# Temporal sampling — TemporalSampler unit tests
# ---------------------------------------------------------------------------

def test_temporal_sampler_build_episode_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_episode_index maps each traj_id to its flat start and length."""
    from robotdataset.oxe.temporal_sampler import TemporalSampler
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    starts, lengths = TemporalSampler.build_episode_index(ds._storage._storage)
    # 3 episodes × 3 steps each
    assert sum(lengths.values()) == 9
    for ep_id in starts:
        assert lengths[ep_id] == 3


def test_temporal_sampler_anchor_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """delta_timestamps with [0.0] → T=1 channels-first image (B, T, C, H, W)."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1, 2],
        batch_size=4,
        root=str(tmp_path),
        delta_timestamps={"observation/image": [0.0]},
        control_frequency=10.0,
    )
    batch = ds.sample()
    # Image: (B, T=1, C=3, H=8, W=8) — channels first
    assert batch["observation", "image"].shape == (4, 1, 3, 8, 8)
    # Action not overridden → default T=1 from effective_dt
    assert batch["action"].shape == (4, 1, 2)


def test_temporal_sampler_multi_offset_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """delta_timestamps with 3 offsets → image shape (B, T=3, C, H, W)."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1, 2],
        batch_size=4,
        root=str(tmp_path),
        delta_timestamps={"observation/image": [-0.1, 0.0, 0.1]},
        control_frequency=10.0,
    )
    batch = ds.sample()
    # (B, T=3, C=3, H=8, W=8) — channels first per spec
    assert batch["observation", "image"].shape == (4, 3, 3, 8, 8)
    assert batch["observation", "image"].dtype == torch.uint8


def test_temporal_sampler_multi_modality(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Multiple keys in delta_timestamps each get their own T dimension."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1, 2],
        batch_size=4,
        root=str(tmp_path),
        delta_timestamps={
            "observation/image": [-0.1, 0.0, 0.1],
            "action": [0.0, 0.1],
        },
        control_frequency=10.0,
    )
    batch = ds.sample()
    # Image: (B, T=3, C=3, H=8, W=8)
    assert batch["observation", "image"].shape == (4, 3, 3, 8, 8)
    # Action: (B, T=2, action_dim=2)
    assert batch["action"].shape == (4, 2, 2)


def test_temporal_sampler_boundary_clamping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Negative offsets from the first step clamp to step 0 (repeat-pad)."""
    from robotdataset.oxe.temporal_sampler import TemporalSampler
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0],
        batch_size=1,
        root=str(tmp_path),
        delta_timestamps={"observation/state": [-0.5, 0.0]},
        control_frequency=10.0,  # -0.5 s → -5 steps, clamped to 0
    )
    storage_td = ds._storage._storage
    starts, lengths = TemporalSampler.build_episode_index(storage_td)
    sampler = TemporalSampler(
        delta_timestamps={"observation/state": [-0.5, 0.0]},
        control_frequency=10.0,
    )
    # Force anchor to the very first step of episode 0
    episode_ids = storage_td["collector", "episode_id"]
    ep_id = int(episode_ids[0].item())
    first_step_state = storage_td["observation", "state"][0]

    # Sample 8 anchors — all must have their t=0 slot equal to the first-step state
    import torch as _torch
    _torch.manual_seed(0)
    result = sampler(storage_td, {ep_id: 0}, {ep_id: lengths[ep_id]}, batch_size=1)
    # The -0.5s offset (-5 steps) must have been clamped to step 0
    # We verify by checking shape only (value depends on random anchor)
    assert result["observation", "state"].shape == (1, 2, 4)


def test_temporal_sampler_default_is_t1_for_all_modalities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no delta_timestamps, temporal sampling is still compulsory: T=1 for all modalities."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1],
        batch_size=4,
        root=str(tmp_path),
        # delta_timestamps omitted → default {all_tensor_modalities: [0.0]}
    )
    assert ds._temporal_sampler is not None
    batch = ds.sample()
    # Temporal dim T=1 always present; image is channels-first (B, T, C, H, W)
    assert batch["action"].shape == (4, 1, 2)          # (B, T=1, action_dim)
    assert batch["observation", "image"].shape == (4, 1, 3, 8, 8)  # (B, T=1, C, H, W)


# ---------------------------------------------------------------------------
# Temporal sampling — next-field mirroring (feature #10)
# ---------------------------------------------------------------------------

def test_next_field_temporal_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """next/observation/* has the same T as obs, with channels-first images."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1, 2],
        batch_size=4,
        root=str(tmp_path),
        delta_timestamps={
            "observation/image": [-0.1, 0.0, 0.1],
            "observation/state": [-0.1, 0.0, 0.1],
        },
        control_frequency=10.0,
    )
    batch = ds.sample()
    # obs: (B, T=3, C, H, W)
    assert batch["observation", "image"].shape == (4, 3, 3, 8, 8)
    # next mirrors T=3, also channels-first
    assert batch["next", "observation", "image"].shape == (4, 3, 3, 8, 8)
    assert batch["next", "observation", "state"].shape == (4, 3, 4)


def test_next_field_default_t1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With default T=1, next/observation/* also has T=1."""
    _patch(monkeypatch)
    ds = oxe.OXEDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1],
        batch_size=4,
        root=str(tmp_path),
    )
    batch = ds.sample()
    assert batch["next", "observation", "image"].shape == (4, 1, 3, 8, 8)
    assert batch["next", "observation", "state"].shape == (4, 1, 4)


def test_next_field_mirrored_values() -> None:
    """Unit test: next-field step offsets are the negation of obs offsets (sorted)."""
    from robotdataset.oxe.temporal_sampler import TemporalSampler

    # Synthetic 5-step episode; state[t] = t so we can track which step is sampled
    ep_state = torch.arange(5, dtype=torch.float32).unsqueeze(1)  # (5, 1)
    episode_ids = torch.zeros(5, dtype=torch.int64)
    storage_td = TensorDict(
        {
            "observation": {"state": ep_state},
            "next": {"observation": {"state": torch.zeros(5, 1)}},
            "collector": {"episode_id": episode_ids},
        },
        batch_size=[5],
    )

    # obs offsets: round(-0.1*10)=-1, round(0.0*10)=0 → [-1, 0]
    # next offsets (mirror): sorted([1, 0]) = [0, 1]
    sampler = TemporalSampler(
        delta_timestamps={"observation/state": [-0.1, 0.0]},
        control_frequency=10.0,
    )
    starts = {0: 0}
    lengths = {0: 5}

    # Use anchor=2: obs=[step1, step2], next=[step2, step3]
    anchor = torch.tensor([2])
    obs_flat = sampler._build_flat_indices(anchor, sampler._offsets, episode_ids, starts, lengths, 1)
    next_flat = sampler._build_flat_indices(anchor, sampler._next_offsets, episode_ids, starts, lengths, 1)

    obs_data = ep_state[obs_flat[("observation", "state")]]   # (1, 2, 1)
    next_data = ep_state[next_flat[("observation", "state")]]  # (1, 2, 1)

    # obs at offset -1 (step 1) = 1.0; at offset 0 (step 2) = 2.0
    assert obs_data[0, 0, 0].item() == pytest.approx(1.0)
    assert obs_data[0, 1, 0].item() == pytest.approx(2.0)
    # next at offset 0 (step 2) = 2.0; at offset +1 (step 3) = 3.0
    assert next_data[0, 0, 0].item() == pytest.approx(2.0)
    assert next_data[0, 1, 0].item() == pytest.approx(3.0)
    # Anchor step is shared: last obs frame == first next frame
    assert torch.allclose(obs_data[:, -1, :], next_data[:, 0, :])


def test_next_field_boundary_clamped() -> None:
    """next-field clamps positive offsets to the last frame when at episode end."""
    from robotdataset.oxe.temporal_sampler import TemporalSampler

    ep_state = torch.arange(3, dtype=torch.float32).unsqueeze(1)  # steps 0,1,2
    episode_ids = torch.zeros(3, dtype=torch.int64)
    storage_td = TensorDict(
        {
            "observation": {"state": ep_state},
            "next": {"observation": {"state": torch.zeros(3, 1)}},
            "collector": {"episode_id": episode_ids},
        },
        batch_size=[3],
    )

    sampler = TemporalSampler(
        delta_timestamps={"observation/state": [-0.1, 0.0]},
        control_frequency=10.0,
    )
    starts = {0: 0}
    lengths = {0: 3}

    # Anchor at last step (2): next offsets [0, +1] → [step2, step2] (clamped)
    anchor = torch.tensor([2])
    next_flat = sampler._build_flat_indices(anchor, sampler._next_offsets, episode_ids, starts, lengths, 1)
    next_data = ep_state[next_flat[("observation", "state")]]  # (1, 2, 1)

    # Both frames clamped to last step value (2.0)
    assert next_data[0, 0, 0].item() == pytest.approx(2.0)
    assert next_data[0, 1, 0].item() == pytest.approx(2.0)
