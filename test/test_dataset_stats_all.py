"""generate_stats() behaves the same on OXE, Table30v2 and AgiBot (faked data layers)."""

from __future__ import annotations

import pytest

pytest.importorskip("torchrl")
pytest.importorskip("datasets", reason="hf extras not installed")
pytest.importorskip("tensorflow", reason="oxe extras not installed")

import robotdataset.dataset_stats as dataset_stats  # noqa: E402
import robotdataset.oxe_dataset as oxe  # noqa: E402
import robotdataset.table30v2_dataset as t30  # noqa: E402
import test_agibot_dataset as agi  # noqa: E402
import test_oxe_dataset as oxe_t  # noqa: E402
import test_table30v2_dataset as t30_t  # noqa: E402
from robotdataset.stats import EpisodeStats  # noqa: E402


def _oxe(mp, root, episodes):
    oxe_t._patch(mp)
    return oxe.OXEDataset(dataset_name="droid", split="train", episodes=episodes, root=str(root), batch_size=2)


def _t30(mp, root, episodes):
    t30_t._patch(mp)
    return t30.Table30v2Dataset(split="train", episodes=episodes, root=str(root), batch_size=2)


def _agibot(mp, root, episodes):
    agi._patch_all(mp)
    return agi.agibot_mod.AgiBotWorldBetaDataset(
        tasks=[agi.TASK_A, agi.TASK_B], cameras=agi.CAMERAS, episodes=episodes, root=str(root), batch_size=2
    )


@pytest.fixture(params=[_oxe, _t30, _agibot], ids=["oxe", "table30v2", "agibot"])
def make(request, monkeypatch, tmp_path):
    def build(episodes, root=None):
        return request.param(monkeypatch, root or tmp_path / "cache", episodes)

    return build


def test_stats_cover_exactly_the_selection(make):
    ds = make([0, 2])
    st = ds.generate_stats()
    assert st, "expected at least one float signal leaf"
    assert {v["count"] for v in st.values()} == {len(ds)}
    assert not any(k.startswith(("next/", "collector/")) for k in st)
    assert ds.statistics == st
    ds.sample()


def test_exact_matches_merged(make):
    import numpy as np

    ds = make([0, 1, 2])
    a, b = ds.generate_stats(), ds.generate_stats(exact=True)
    assert a.keys() == b.keys()
    for k in a:
        for s in ("mean", "std", "min", "max", "q01", "q99"):
            np.testing.assert_allclose(a[k][s], b[k][s], rtol=1e-9, atol=1e-9)


def test_growth_computes_only_new_episodes(make, monkeypatch):
    make([0, 1])
    calls = []
    real = dataset_stats.compute_episode_stats
    monkeypatch.setattr(
        dataset_stats, "compute_episode_stats", lambda td, **kw: (calls.append(kw["seed"]), real(td, **kw))[1]
    )
    ds = make([0, 1, 2, 3])
    assert len(calls) == 2


def test_backfill_without_rebuild(make):
    ds = make([0, 1])
    for i in (0, 1):
        for f in ("stats.json", "stats_sample.npz"):
            (ds.data_path / str(i) / f).unlink()
    stamps = {i: (ds.data_path / str(i) / "_steps.json").stat().st_mtime_ns for i in (0, 1)}
    ds2 = make([0, 1])
    assert all(EpisodeStats.load(ds2.data_path / str(i)) is not None for i in (0, 1))
    assert stamps == {i: (ds2.data_path / str(i) / "_steps.json").stat().st_mtime_ns for i in (0, 1)}


def test_keys_subset_and_unknown(make):
    ds = make([0, 1])
    first = sorted(ds.generate_stats())[0]
    assert list(ds.generate_stats(keys=[first])) == [first]
    with pytest.raises(ValueError, match="valid keys"):
        ds.generate_stats(keys=["nope"])


def test_non_finite_does_not_block_construction(make, monkeypatch):
    real = dataset_stats.compute_episode_stats

    def boom(td, **kw):
        if kw["seed"] == 1:
            raise ValueError("non-finite")
        return real(td, **kw)

    monkeypatch.setattr(dataset_stats, "compute_episode_stats", boom)
    with pytest.warns(UserWarning, match="episode 1"):
        ds = make([0, 1])
    ds.sample()
    with pytest.raises(ValueError, match="episode 1"):
        ds.generate_stats()
