"""generate_stats() / statistics through LeRobotVideoDataset (offline synthetic tree)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")
pytest.importorskip("imageio")
pytest.importorskip("pandas")

import robotdataset.dataset_stats as dataset_stats  # noqa: E402
from robotdataset.lerobot_video_dataset import LeRobotVideoDataset  # noqa: E402
from robotdataset.stats import EpisodeStats  # noqa: E402
from test_lerobot_video_dataset import make_dataset  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    return make_dataset(tmp_path / "ds", episode_lengths=(6, 4, 5, 7, 3))


def _ds(tree, tmp_path, episodes):
    return LeRobotVideoDataset(
        local_path=str(tree), episodes=episodes, root=str(tmp_path / "cache"), batch_size=2
    )


def test_stats_match_numpy(tree, tmp_path):
    ds = _ds(tree, tmp_path, [0, 1])
    st = ds.generate_stats()
    assert "action" in st and "observation/state/joints" in st
    assert not any(k.startswith(("next/", "collector/", "observation/images")) for k in st)
    act = np.concatenate([[[t * 0.1, -t * 0.1] for t in range(n)] for n in (6, 4)])
    assert st["action"]["count"] == 10
    np.testing.assert_allclose(st["action"]["mean"], act.mean(0), atol=1e-6)
    np.testing.assert_allclose(st["action"]["std"], act.std(0), atol=1e-6)
    assert ds.statistics == st
    ds.sample()  # still works


def test_exact_matches_merged(tree, tmp_path):
    ds = _ds(tree, tmp_path, [0, 1, 2])
    a, b = ds.generate_stats(), ds.generate_stats(exact=True)
    for k in a:
        for s in ("mean", "std", "min", "max"):
            np.testing.assert_allclose(a[k][s], b[k][s], rtol=1e-9, atol=1e-9)
        assert a[k]["count"] == b[k]["count"]
        # samples cover every row (episodes shorter than sample_size), so quantiles agree too
        np.testing.assert_allclose(a[k]["q01"], b[k]["q01"], atol=1e-9)
        np.testing.assert_allclose(a[k]["q99"], b[k]["q99"], atol=1e-9)


def test_progressive_growth_computes_only_new_episodes(tree, tmp_path, monkeypatch):
    _ds(tree, tmp_path, [0, 1, 2])
    calls = []
    real = dataset_stats.compute_episode_stats
    monkeypatch.setattr(
        dataset_stats, "compute_episode_stats", lambda td, **kw: (calls.append(kw["seed"]), real(td, **kw))[1]
    )
    ds = _ds(tree, tmp_path, [0, 1, 2, 3, 4])
    assert sorted(calls) == [3, 4]
    fresh = _ds(tree, tmp_path / "other", [0, 1, 2, 3, 4]).generate_stats()
    assert ds.generate_stats() == fresh


def test_backfill_without_rebuild(tree, tmp_path):
    ds = _ds(tree, tmp_path, [0, 1])
    for i in (0, 1):
        for f in ("stats.json", "stats_sample.npz"):
            (ds.data_path / str(i) / f).unlink()
    mtimes = {i: (ds.data_path / str(i) / "_steps.json").stat().st_mtime_ns for i in (0, 1)}
    ds2 = _ds(tree, tmp_path, [0, 1])
    assert EpisodeStats.load(ds.data_path / "0") is not None
    assert mtimes == {i: (ds2.data_path / str(i) / "_steps.json").stat().st_mtime_ns for i in (0, 1)}


def test_subset_and_keys(tree, tmp_path):
    ds = _ds(tree, tmp_path, [2, 4])
    assert ds.generate_stats()["action"]["count"] == 8
    assert list(ds.generate_stats(keys=["action"])) == ["action"]
    with pytest.raises(ValueError, match="valid keys"):
        ds.generate_stats(keys=["nope"])
    with pytest.raises(ValueError, match="valid keys"):
        ds.generate_stats(keys=["nope"], exact=True)


def test_force_recomputes(tree, tmp_path, monkeypatch):
    ds = _ds(tree, tmp_path, [0, 1])
    calls = []
    real = dataset_stats.compute_episode_stats
    monkeypatch.setattr(
        dataset_stats, "compute_episode_stats", lambda td, **kw: (calls.append(1), real(td, **kw))[1]
    )
    ds.generate_stats()
    assert not calls
    ds.generate_stats(force=True)
    assert len(calls) == 2


def test_non_finite_does_not_block_construction(tree, tmp_path, monkeypatch):
    real = dataset_stats.compute_episode_stats

    def boom(td, **kw):
        if kw["seed"] == 1:
            raise ValueError("Feature 'action' has 1 row(s) with non-finite values (NaN/inf)")
        return real(td, **kw)

    monkeypatch.setattr(dataset_stats, "compute_episode_stats", boom)
    with pytest.warns(UserWarning, match="episode 1"):
        ds = _ds(tree, tmp_path, [0, 1])
    ds.sample()
    with pytest.raises(ValueError, match="episode 1"):
        ds.generate_stats()
