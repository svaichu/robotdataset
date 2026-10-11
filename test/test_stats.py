"""Tests for robotdataset.stats (per-episode records, merge) and robotdataset.gr00t."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tensordict = pytest.importorskip("tensordict")
from tensordict import TensorDict  # noqa: E402

from robotdataset.gr00t import check_against, to_gr00t_statistics  # noqa: E402
from robotdataset.stats import (  # noqa: E402
    EpisodeStats,
    compute_episode_stats,
    merge_episode_stats,
    signal_leaves,
    weighted_quantile,
)

LENGTHS = (5, 300, 40, 1000)


def make_td(n: int, rng, action_dim: int = 4, nan_at=None) -> TensorDict:
    action = torch.from_numpy(rng.normal(size=(n, action_dim)).astype(np.float32) * 3 + 1)
    if nan_at is not None:
        action[nan_at, 0] = float("nan")
    return TensorDict(
        {
            "action": action,
            "observation": TensorDict(
                {
                    "state": TensorDict({"joints": torch.from_numpy(rng.normal(size=(n, 3)).astype(np.float32))}, [n]),
                    "images": TensorDict({"top": torch.zeros(n, 4, 4, 3)}, [n]),
                },
                [n],
            ),
            "done": torch.zeros(n, 1, dtype=torch.bool),
            "next": TensorDict({"reward": torch.zeros(n, 1), "obs": torch.zeros(n, 2)}, [n]),
            "collector": TensorDict({"episode_id": torch.zeros(n, dtype=torch.int64)}, [n]),
        },
        [n],
    )


@pytest.fixture
def episodes():
    rng = np.random.default_rng(0)
    return [make_td(n, rng) for n in LENGTHS]


def _cat(episodes, path):
    return np.concatenate(
        [e.get(tuple(path.split("/"))).reshape(len(e), -1).double().numpy() for e in episodes]
    )


def test_default_keys_exclude_noise(episodes):
    assert sorted(signal_leaves(episodes[0])) == ["action", "observation/state/joints"]


def test_merge_moments_match_numpy(episodes):
    recs = [compute_episode_stats(e, sample_size=64, seed=i) for i, e in enumerate(episodes)]
    stats = merge_episode_stats(recs)
    for path in ("action", "observation/state/joints"):
        ref = _cat(episodes, path)
        s = stats[path]
        assert s["count"] == len(ref)
        np.testing.assert_allclose(s["mean"], ref.mean(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s["std"], ref.std(0), rtol=1e-10)
        np.testing.assert_array_equal(s["min"], ref.min(0))
        np.testing.assert_array_equal(s["max"], ref.max(0))


def test_single_record_merge_matches_itself(episodes):
    rec = compute_episode_stats(episodes[1], sample_size=10_000)
    s = merge_episode_stats([rec])["action"]
    ref = episodes[1]["action"].double().numpy()
    np.testing.assert_allclose(s["mean"], ref.mean(0))
    np.testing.assert_allclose(s["std"], ref.std(0))


def test_quantiles_exact_when_sample_covers_all(episodes):
    recs = [compute_episode_stats(e, sample_size=10_000) for e in episodes]
    s = merge_episode_stats(recs)["action"]
    ref = _cat(episodes, "action")
    np.testing.assert_allclose(s["q01"], np.quantile(ref, 0.01, axis=0), rtol=1e-9)
    np.testing.assert_allclose(s["q99"], np.quantile(ref, 0.99, axis=0), rtol=1e-9)


def test_weighted_quantile_reduces_to_numpy_linear():
    v = np.random.default_rng(1).normal(size=97)
    got = weighted_quantile(v, np.full(97, 3.0), (0.01, 0.5, 0.99))
    np.testing.assert_allclose(got, np.quantile(v, [0.01, 0.5, 0.99]))


def test_subsampled_quantile_error_on_heavy_tails():
    rng = np.random.default_rng(2)
    eps = [TensorDict({"action": torch.from_numpy(rng.standard_t(3, size=(n, 1)).astype(np.float32))}, [n])
           for n in (2000, 3000, 1500)]
    ref = _cat(eps, "action")
    recs = [compute_episode_stats(e, sample_size=256, seed=i) for i, e in enumerate(eps)]
    s = merge_episode_stats(recs)["action"]
    err = max(abs(s["q01"][0] - np.quantile(ref, 0.01)), abs(s["q99"][0] - np.quantile(ref, 0.99)))
    # Documented bound: within 0.5 std on t(3) data with 256 samples/episode.
    assert err < 0.5 * ref.std()


def test_seeding(episodes):
    a = compute_episode_stats(episodes[1], sample_size=32, seed=7)
    b = compute_episode_stats(episodes[1], sample_size=32, seed=7)
    c = compute_episode_stats(episodes[1], sample_size=32, seed=8)
    np.testing.assert_array_equal(a.sample["action"], b.sample["action"])
    assert not np.array_equal(a.sample["action"], c.sample["action"])


def test_short_episode_keeps_all_rows(episodes):
    rec = compute_episode_stats(episodes[0], sample_size=256)
    assert rec.sample["action"].shape == (5, 4)


def test_save_load_roundtrip_and_invalid(tmp_path, episodes):
    rec = compute_episode_stats(episodes[1], sample_size=32, seed=1)
    rec.save(tmp_path)
    got = EpisodeStats.load(tmp_path)
    assert got.moments == rec.moments and got.schema == rec.schema and got.num_rows == rec.num_rows
    np.testing.assert_array_equal(got.sample["observation/state/joints"], rec.sample["observation/state/joints"])
    assert not list(tmp_path.glob("*.tmp"))

    meta = json.loads((tmp_path / "stats.json").read_text())
    meta["version"] = 999
    (tmp_path / "stats.json").write_text(json.dumps(meta))
    assert EpisodeStats.load(tmp_path) is None
    (tmp_path / "stats.json").write_text("{not json")
    assert EpisodeStats.load(tmp_path) is None
    assert EpisodeStats.load(tmp_path / "missing") is None


def test_non_finite_raises_with_feature_name():
    td = make_td(10, np.random.default_rng(0), nan_at=3)
    with pytest.raises(ValueError, match=r"'action'.*1 row"):
        compute_episode_stats(td)


def test_merge_schema_mismatch_and_unknown_key(episodes):
    rng = np.random.default_rng(0)
    a = compute_episode_stats(episodes[0])
    b = compute_episode_stats(make_td(10, rng, action_dim=6))
    with pytest.raises(ValueError, match="schemas differ"):
        merge_episode_stats([a, b])
    with pytest.raises(ValueError, match="valid keys"):
        merge_episode_stats([a], keys=["nope"])
    assert list(merge_episode_stats([a], keys=["action"])) == ["action"]


def test_memmap_directory_ignores_extra_stats_files(tmp_path, episodes):
    d = tmp_path / "ep"
    d.mkdir()
    episodes[1].memmap_(str(d))
    compute_episode_stats(TensorDict.load_memmap(str(d))).save(d)
    td = TensorDict.load_memmap(str(d))
    assert len(td) == LENGTHS[1]
    TensorDict.load_memmap(str(d))[:1].expand(3).memmap_like(str(tmp_path / "c"))


# ---------------------------------------------------------------------------
# GR00T export
# ---------------------------------------------------------------------------

def _stats():
    rng = np.random.default_rng(0)
    eps = [make_td(50, rng, action_dim=8) for _ in range(3)]
    st = merge_episode_stats([compute_episode_stats(e, seed=i) for i, e in enumerate(eps)])
    st["observation/state/all"] = st["observation/state/joints"]
    return st


def test_gr00t_layout():
    g = to_gr00t_statistics(_stats(), "emb", action_groups={"arm": (0, 7), "gripper": (7, 8)})
    assert set(g) == {"emb"}
    assert set(g["emb"]["state"]) == {"joints"}  # "all" dropped... only one slice here
    assert set(g["emb"]["action"]) == {"arm", "gripper"}
    for grp in (*g["emb"]["state"].values(), *g["emb"]["action"].values()):
        assert set(grp) == {"mean", "std", "min", "max", "q01", "q99"}
        assert len({len(v) for v in grp.values()}) == 1
    assert len(g["emb"]["action"]["arm"]["mean"]) == 7


def test_gr00t_modality_meta_equivalent():
    st = _stats()
    a = to_gr00t_statistics(st, "e", action_groups={"arm": (0, 7), "gripper": (7, 8)})
    b = to_gr00t_statistics(
        st, "e", modality_meta={"action": {"arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}}}
    )
    assert a == b


def test_gr00t_errors():
    st = _stats()
    with pytest.raises(ValueError, match="'arm'"):
        to_gr00t_statistics(st, "e", action_groups={"arm": (0, 9)})
    with pytest.raises(ValueError, match="'arm'"):
        to_gr00t_statistics(st, "e", action_groups={"arm": (4, 4)})
    with pytest.raises(ValueError, match="missing"):
        to_gr00t_statistics(st, "e", action_groups={"a": (0, 1)}, action_key="nope")
    with pytest.raises(ValueError, match="missing"):
        to_gr00t_statistics(st, "e", state_groups={"j": "observation/state/zzz"}, action_groups={"a": (0, 1)})


def test_gr00t_check_against():
    g = to_gr00t_statistics(_stats(), "e", action_groups={"arm": (0, 8)})
    check_against(g, {"state": ["state.joints"], "action": {"modality_keys": ["action.arm"]}})
    with pytest.raises(ValueError, match="action"):
        check_against(g, {"action": ["action.other"]})
