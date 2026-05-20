from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax is required for OXEJAXDataset tests")
import jax.numpy as jnp

import robotdataset.oxe_jax_dataset as oxe_jax


_FAKE_CACHE = {"droid": {"1.0.1": "gs://gresearch/robotics/droid/1.0.1"}}


def _make_episode(episode_idx: int, n_steps: int = 3) -> dict:
    steps = []
    for t in range(n_steps):
        steps.append(
            {
                "observation": {
                    "image": np.full((8, 8, 3), t, dtype=np.uint8),
                    "state": np.array([float(episode_idx), float(t), 1.0, 2.0], dtype=np.float32),
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

    def __iter__(self):
        return iter(self._items)


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


def _patch(monkeypatch: pytest.MonkeyPatch, episodes=None) -> None:
    eps = episodes if episodes is not None else _EPISODES
    fake_tf = SimpleNamespace(Tensor=np.ndarray, io=SimpleNamespace(gfile=SimpleNamespace()))
    fake_tfds = SimpleNamespace(builder_from_directory=lambda **kw: FakeBuilder(eps))
    monkeypatch.setattr(oxe_jax, "tf", fake_tf, raising=False)
    monkeypatch.setattr(oxe_jax, "tfds", fake_tfds, raising=False)
    monkeypatch.setattr(oxe_jax, "_TF_TENSOR_TYPES", tuple(), raising=False)
    monkeypatch.setattr(oxe_jax, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)
    monkeypatch.setattr(oxe_jax, "_copy_tree", lambda src, dst: None)
    monkeypatch.setattr(oxe_jax, "_copy_metadata_only", lambda src, dst: None)


def test_jax_episode_filter_and_lengths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe_jax.OXEJAXDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1],
        root=str(tmp_path),
        batch_size=4,
    )
    assert ds.num_episodes == 2
    assert len(ds) == 6


def test_jax_cache_reuse_for_same_episode_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    calls = []
    original = oxe_jax.build_missing_episodes

    def spy(*args, **kwargs):
        calls.append(kwargs.get("missing", args[3] if len(args) > 3 else []))
        return original(*args, **kwargs)

    monkeypatch.setattr(oxe_jax, "build_missing_episodes", spy)
    oxe_jax.OXEJAXDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert calls == [[0, 1, 2]]

    calls.clear()
    oxe_jax.OXEJAXDataset(dataset_name="droid", split="train", episodes=[0, 1, 2], root=str(tmp_path))
    assert calls == []


def test_jax_modalities_inference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe_jax.OXEJAXDataset(dataset_name="droid", split="train", root=str(tmp_path))
    mods = ds.modalities
    assert "observation/image" in mods["image"]
    assert "action" in mods["action"]
    assert "language_instruction" in mods["text"]


def test_jax_temporal_shapes_and_types(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe_jax.OXEJAXDataset(
        dataset_name="droid",
        split="train",
        episodes=[0, 1, 2],
        batch_size=3,
        root=str(tmp_path),
        delta_timestamps={
            "observation/image": [-0.1, 0.0, 0.1],
            "observation/state": [-0.1, 0.0, 0.1],
            "action": [0.0, 0.1],
        },
        control_frequency=10.0,
    )
    batch = ds.sample(rng=np.random.default_rng(0))

    assert isinstance(batch["observation"]["image"], jax.Array)
    assert isinstance(batch["action"], jax.Array)
    assert batch["observation"]["image"].shape == (3, 3, 3, 8, 8)
    assert batch["next"]["observation"]["image"].shape == (3, 3, 3, 8, 8)
    assert batch["observation"]["state"].shape == (3, 3, 4)
    assert batch["action"].shape == (3, 2, 2)
    assert batch["observation"]["image"].dtype == jnp.uint8
    assert batch["action"].dtype == jnp.float32


def test_jax_text_is_non_array_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe_jax.OXEJAXDataset(dataset_name="droid", split="train", episodes=[0], root=str(tmp_path), batch_size=2)
    batch = ds.sample(rng=np.random.default_rng(1))
    assert isinstance(batch["language_instruction"], list)
    assert all(isinstance(item, str) for item in batch["language_instruction"])


def test_jax_next_mirrored_window_values() -> None:
    store = oxe_jax.CombinedNumpyStore.__new__(oxe_jax.CombinedNumpyStore)  # bypass file loading
    store.n_steps = 5
    ep_state = np.arange(5, dtype=np.float32)[:, None]
    store.leaves = {
        ("observation", "state"): ep_state,
        ("next", "observation", "state"): np.zeros((5, 1), dtype=np.float32),
        ("collector", "episode_id"): np.zeros((5,), dtype=np.int64),
    }

    sampler = oxe_jax.JAXTemporalSampler(
        delta_timestamps={"observation/state": [-0.1, 0.0]},
        control_frequency=10.0,
    )
    starts = {0: 0}
    lengths = {0: 5}

    anchor = np.array([2], dtype=np.int64)
    obs_flat = sampler._build_flat_indices(anchor, sampler._offsets, store.leaves[("collector", "episode_id")], starts, lengths)
    next_flat = sampler._build_flat_indices(anchor, sampler._next_offsets, store.leaves[("collector", "episode_id")], starts, lengths)
    obs_data = ep_state[obs_flat[("observation", "state")]]
    next_data = ep_state[next_flat[("observation", "state")]]

    assert obs_data[0, 0, 0] == pytest.approx(1.0)
    assert obs_data[0, 1, 0] == pytest.approx(2.0)
    assert next_data[0, 0, 0] == pytest.approx(2.0)
    assert next_data[0, 1, 0] == pytest.approx(3.0)
