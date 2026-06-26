from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import robotdataset.libero_dataset as libero
from robotdataset.hf.loader import get_episode_ids


N_EPISODES = 4
N_STEPS = 3


def _make_episode_row(episode_idx: int, n_steps: int = N_STEPS) -> dict:
    task_id = episode_idx % 2
    steps = []
    for t in range(n_steps):
        steps.append(
            {
                "observation": {
                    "image": np.zeros((8, 8, 3), dtype=np.uint8),
                    "state": np.array([episode_idx, t], dtype=np.float32),
                },
                "action": np.array([episode_idx, t], dtype=np.float32),
                "reward": float(t),
                "is_last": bool(t == n_steps - 1),
                "is_terminal": bool(t == n_steps - 1),
            }
        )
    return {"steps": steps, "task_id": task_id}


class FakeEpisodeDataset:
    def __init__(self, rows: list):
        self._rows = list(rows)
        self.features = {"steps": None, "task_id": None}

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[key]

    def __len__(self):
        return len(self._rows)


def _patch(monkeypatch: pytest.MonkeyPatch):
    rows = [_make_episode_row(i) for i in range(N_EPISODES)]
    fake = FakeEpisodeDataset(rows)
    monkeypatch.setattr(libero, "load_hf_dataset", lambda *a, **kw: fake)
    return fake


def test_get_episode_ids_fallback_to_row_indices(monkeypatch):
    fake = _patch(monkeypatch)
    assert get_episode_ids(fake) == [0, 1, 2, 3]


def test_libero_dataset_step_count(monkeypatch, tmp_path: Path):
    _patch(monkeypatch)
    ds = libero.LiberoDataset(root=str(tmp_path))
    assert len(ds) == N_EPISODES * N_STEPS
    assert ds.num_episodes == N_EPISODES


def test_libero_sample_td_keys(monkeypatch, tmp_path: Path):
    _patch(monkeypatch)
    ds = libero.LiberoDataset(batch_size=2, root=str(tmp_path))
    batch = ds.sample()
    assert isinstance(batch, TensorDict)
    assert "observation" in batch.keys()
    assert "action" in batch.keys()
    assert "next" in batch.keys()
    assert "reward" in batch["next"].keys()


def test_libero_image_channels_first(monkeypatch, tmp_path: Path):
    _patch(monkeypatch)
    ds = libero.LiberoDataset(batch_size=2, root=str(tmp_path))
    batch = ds.sample()
    assert batch["observation", "image"].shape == (2, 1, 3, 8, 8)
    assert batch["observation", "image"].dtype == torch.uint8


def test_libero_tasks_filter(monkeypatch, tmp_path: Path):
    _patch(monkeypatch)
    ds = libero.LiberoDataset(tasks=[1], root=str(tmp_path))
    assert ds.num_episodes == 2
    assert len(ds) == 2 * N_STEPS


def test_libero_config_name_changes_cache_path(monkeypatch, tmp_path: Path):
    _patch(monkeypatch)
    ds = libero.LiberoDataset(config_name="libero_goal", root=str(tmp_path))
    assert ds.data_path == tmp_path / "hf" / "libero" / "train" / "libero_goal" / "episodes"
