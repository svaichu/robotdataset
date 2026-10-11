"""Relative-action statistics: geometry, per-episode records, merge, dataset method (offline)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import robotdataset.dataset_stats as dataset_stats  # noqa: E402
from robotdataset.dataset_stats import DatasetStatsMixin  # noqa: E402
from robotdataset.gr00t import to_gr00t_statistics  # noqa: E402
from robotdataset.relative_stats import (  # noqa: E402
    RelativeSpec,
    build_specs,
    compute_relative_episode_stats,
    matrix_to_rotvec,
    relative_chunks,
    rotvec_to_matrix,
)
from robotdataset.stats import EpisodeStats  # noqa: E402

STATS = ("mean", "std", "min", "max", "q01", "q99")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pose_episode(n, seed):
    """Random xyz+rotvec action and state (rotations up to ~2.5 rad)."""
    rng = np.random.default_rng(seed)
    pose = lambda: np.concatenate([rng.normal(size=(n, 3)), rng.normal(size=(n, 3)) * 1.2], axis=1)
    return pose(), pose()


def _scipy_relative(actions, states, horizon):
    """Reference chunks built pose by pose with scipy (what GR00T's pose classes do)."""
    from scipy.spatial.transform import Rotation

    def homogeneous(v):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(v[3:]).as_matrix()
        t[:3, 3] = v[:3]
        return t

    out = []
    for i in range(len(actions) - horizon + 1):
        inv = np.linalg.inv(homogeneous(states[i]))
        chunk = []
        for h in range(horizon):
            rel = inv @ homogeneous(actions[i + h])
            chunk.append(np.concatenate([rel[:3, 3], Rotation.from_matrix(rel[:3, :3]).as_rotvec()]))
        out.append(chunk)
    return np.asarray(out)


class Host(DatasetStatsMixin):
    """Minimal dataset: episode memmaps under ``root/<i>``."""

    def __init__(self, root, loaded):
        self.root = root
        self._loaded_indices = list(loaded)

    def _episodes_dir(self):
        return self.root


def _write_episodes(root, lengths, dim=2, seed=0, pose=False, state_dim=None):
    from tensordict import TensorDict

    rng = np.random.default_rng(seed)
    store = {}
    for i, n in enumerate(lengths):
        if pose:
            a, s = _pose_episode(n, seed + i)
        else:
            a, s = rng.normal(size=(n, dim)), rng.normal(size=(n, state_dim or dim))
        store[i] = (a, s)
        td = TensorDict(
            {"action": torch.tensor(a, dtype=torch.float32),
             "observation": {"state": {"q": torch.tensor(s, dtype=torch.float32)}}},
            batch_size=[n],
        )
        (root / str(i)).mkdir(parents=True)
        td.memmap_(str(root / str(i)))
    return store


def _f32(x):  # what the memmap holds
    return np.asarray(x, dtype=np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_rotation_conversions_match_scipy():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    axes = rng.normal(size=(400, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.concatenate([[0, 1e-12, 1e-6, 9e-5, 1e-4, 1.1e-4, 1e-3, 2e-3], rng.uniform(0, 3.1, 392)])
    rv = axes * angles[:, None]
    mats = rotvec_to_matrix(rv)
    np.testing.assert_allclose(mats, Rotation.from_rotvec(rv).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(matrix_to_rotvec(mats), Rotation.from_matrix(mats).as_rotvec(), atol=1e-9)
    np.testing.assert_allclose(matrix_to_rotvec(mats), rv, atol=1e-9)


def test_matrix_to_rotvec_near_pi_round_trips():
    rng = np.random.default_rng(1)
    axes = rng.normal(size=(200, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    rv = axes * (np.pi - rng.uniform(0, 1e-3, 200))[:, None]
    mats = rotvec_to_matrix(rv)
    back = matrix_to_rotvec(mats)
    np.testing.assert_allclose(rotvec_to_matrix(back), mats, atol=1e-9)
    assert (np.linalg.norm(back, axis=1) <= np.pi + 1e-9).all()


def test_joint_chunks_are_differences():
    rng = np.random.default_rng(2)
    a, s = rng.normal(size=(9, 3)), rng.normal(size=(9, 3))
    spec = RelativeSpec("j", (0, 3), "s", "joint", 4)
    chunks = relative_chunks(a, s, spec)
    assert chunks.shape == (6, 4, 3)  # n - horizon + 1
    for i in range(6):
        np.testing.assert_allclose(chunks[i], a[i : i + 4] - s[i])


def test_eef_chunks_match_scipy_reference():
    a, s = _pose_episode(14, 3)
    spec = RelativeSpec("eef", (0, 6), "s", "xyz_rotvec", 5)
    got = relative_chunks(a, s, spec)
    np.testing.assert_allclose(got, _scipy_relative(a, s, 5), atol=1e-9)


def test_eef_chunk_of_the_reference_itself_is_identity():
    a, _ = _pose_episode(10, 4)
    spec = RelativeSpec("eef", (0, 6), "s", "xyz_rotvec", 3)
    chunks = relative_chunks(a, a, spec)  # reference state == first action of every chunk
    np.testing.assert_allclose(chunks[:, 0], 0.0, atol=1e-12)


def test_action_slice_selects_columns():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(8, 7))
    s = rng.normal(size=(8, 6))
    spec = RelativeSpec("eef", (0, 6), "s", "xyz_rotvec", 2)
    np.testing.assert_allclose(relative_chunks(a, s, spec), relative_chunks(a[:, :6], s, spec))


def test_episode_shorter_than_horizon_has_no_chunks():
    spec = RelativeSpec("j", (0, 2), "s", "joint", 5)
    assert relative_chunks(np.zeros((4, 2)), np.zeros((4, 2)), spec).shape == (0, 5, 2)
    assert spec.expected_rows(4) == 0 and spec.expected_rows(5) == 1


# ---------------------------------------------------------------------------
# spec validation
# ---------------------------------------------------------------------------

def test_spec_validation():
    with pytest.raises(ValueError, match="representation"):
        RelativeSpec("g", (0, 2), "s", "quat", 4)
    with pytest.raises(ValueError, match="6 columns"):
        RelativeSpec("g", (0, 3), "s", "xyz_rotvec", 4)
    with pytest.raises(ValueError, match="horizon"):
        RelativeSpec("g", (0, 2), "s", "joint", 0)
    with pytest.raises(ValueError, match="invalid action range"):
        RelativeSpec("g", (2, 2), "s", "joint", 4)
    with pytest.raises(ValueError, match="state_keys"):
        build_specs({"a": (0, 2), "b": (2, 3)}, {"a": "s"}, 4)
    assert RelativeSpec("g", (0, 2), "s", "joint", 4).stem != RelativeSpec("g", (0, 2), "s", "joint", 5).stem
    assert RelativeSpec("g", (0, 2), "s", "joint", 4).stem == RelativeSpec("g", (0, 2), "s", "joint", 4).stem


# ---------------------------------------------------------------------------
# per-episode record + dataset method
# ---------------------------------------------------------------------------

LENGTHS = (9, 12, 7)
SPEC_KW = dict(groups={"g": (0, 2)}, state_keys={"g": "observation/state/q"}, horizon=4)


def _truth(store, loaded, horizon, pose=False, dim=2):
    spec = RelativeSpec("g", (0, 6 if pose else dim), "s", "xyz_rotvec" if pose else "joint", horizon)
    chunks = np.concatenate([relative_chunks(_f32(store[i][0]), _f32(store[i][1]), spec) for i in loaded])
    return chunks.reshape(len(chunks), -1)


def _assert_stats(got, flat, horizon, dim, quantile_atol=1e-9):
    for k, ref in {"mean": flat.mean(0), "std": flat.std(0), "min": flat.min(0), "max": flat.max(0)}.items():
        np.testing.assert_allclose(np.asarray(got[k]).reshape(-1), ref, atol=1e-9, err_msg=k)
    q = np.quantile(flat, [0.01, 0.99], axis=0)
    np.testing.assert_allclose(np.asarray(got["q01"]).reshape(-1), q[0], atol=quantile_atol)
    np.testing.assert_allclose(np.asarray(got["q99"]).reshape(-1), q[1], atol=quantile_atol)
    assert np.asarray(got["mean"]).shape == (horizon, dim)
    assert got["count"] == len(flat)


def test_joint_stats_match_numpy_and_skip_short_episodes(tmp_path):
    store = _write_episodes(tmp_path, (9, 3, 7))  # episode 1 is shorter than the horizon
    ds = Host(tmp_path, [0, 1, 2])
    got = ds.generate_relative_action_stats(**SPEC_KW)["g"]
    _assert_stats(got, _truth(store, [0, 2], 4), 4, 2)
    assert got["count"] == 6 + 4  # (9-3) + (7-3)
    assert not list((tmp_path / "1").glob("stats_rel_*"))


def test_eef_stats_match_scipy_reference(tmp_path):
    store = _write_episodes(tmp_path, LENGTHS, pose=True)
    ds = Host(tmp_path, [0, 1, 2])
    kw = dict(groups={"g": (0, 6)}, state_keys={"g": "observation/state/q"}, horizon=3, representation="xyz_rotvec")
    got = ds.generate_relative_action_stats(**kw)["g"]
    ref = np.concatenate(
        [_scipy_relative(_f32(store[i][0]), _f32(store[i][1]), 3) for i in range(3)]
    )
    _assert_stats(got, ref.reshape(len(ref), -1), 3, 6, quantile_atol=1e-8)
    exact = ds.generate_relative_action_stats(exact=True, **kw)["g"]
    for k in STATS:
        np.testing.assert_allclose(exact[k], got[k], atol=1e-8)


def test_exact_equals_merged_when_samples_cover_all_chunks(tmp_path):
    _write_episodes(tmp_path, LENGTHS)
    ds = Host(tmp_path, [0, 1, 2])
    merged = ds.generate_relative_action_stats(**SPEC_KW)["g"]
    exact = ds.generate_relative_action_stats(exact=True, **SPEC_KW)["g"]
    for k in STATS:
        np.testing.assert_allclose(merged[k], exact[k], atol=1e-9)
    assert merged["count"] == exact["count"]


def test_sampled_quantiles_are_close_for_long_episodes(tmp_path):
    store = _write_episodes(tmp_path, (900, 700), dim=2)
    ds = Host(tmp_path, [0, 1])
    ds.stats_sample_size = 512  # ~1.1k of the 1.6k chunks; the q01/q99 noise is then ~0.1 std
    est = ds.generate_relative_action_stats(**SPEC_KW)["g"]
    exact = ds.generate_relative_action_stats(exact=True, **SPEC_KW)["g"]
    flat = _truth(store, [0, 1], 4)
    _assert_stats(exact, flat, 4, 2)  # exact path equals np.quantile
    std = np.asarray(exact["std"])
    for k in ("q01", "q99"):
        assert (np.abs(np.asarray(est[k]) - np.asarray(exact[k])) / std).max() < 0.3
    for k in ("mean", "std", "min", "max"):  # moments stay exact
        np.testing.assert_allclose(est[k], exact[k], atol=1e-9)


def test_records_are_cached_on_disk_and_reused(tmp_path, monkeypatch):
    _write_episodes(tmp_path, LENGTHS)
    ds = Host(tmp_path, [0, 1, 2])
    first = ds.generate_relative_action_stats(**SPEC_KW)
    spec = build_specs(**{"groups": SPEC_KW["groups"], "state_keys": SPEC_KW["state_keys"], "horizon": 4})[0]
    for i in range(3):
        assert (tmp_path / str(i) / f"{spec.stem}.json").exists()
        assert (tmp_path / str(i) / f"{spec.stem}_sample.npz").exists()
        assert EpisodeStats.load(tmp_path / str(i)) is None  # the absolute record is a separate file
    calls = []
    real = dataset_stats.compute_relative_episode_stats
    monkeypatch.setattr(
        dataset_stats, "compute_relative_episode_stats",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    assert ds.generate_relative_action_stats(**SPEC_KW) == first  # in-memory
    assert Host(tmp_path, [0, 1, 2]).generate_relative_action_stats(**SPEC_KW) == first  # from disk
    assert not calls
    Host(tmp_path, [0, 1, 2]).generate_relative_action_stats(force=True, **SPEC_KW)
    assert len(calls) == 3
    Host(tmp_path, [0, 1, 2]).generate_relative_action_stats(**{**SPEC_KW, "horizon": 3})  # new fingerprint
    assert len(calls) == 6


def test_progressive_subset_computes_only_new_episodes(tmp_path, monkeypatch):
    _write_episodes(tmp_path, LENGTHS)
    Host(tmp_path, [0, 1]).generate_relative_action_stats(**SPEC_KW)
    seeds = []
    real = dataset_stats.compute_relative_episode_stats
    monkeypatch.setattr(
        dataset_stats, "compute_relative_episode_stats",
        lambda td, spec, sample_size, seed: (seeds.append(seed), real(td, spec, sample_size, seed))[1],
    )
    Host(tmp_path, [0, 1, 2]).generate_relative_action_stats(**SPEC_KW)
    assert seeds == [2]


def test_stale_record_is_recomputed(tmp_path):
    _write_episodes(tmp_path, LENGTHS)
    ds = Host(tmp_path, [0, 1, 2])
    spec = build_specs(SPEC_KW["groups"], SPEC_KW["state_keys"], 4)[0]
    ds.generate_relative_action_stats(**SPEC_KW)
    rec = EpisodeStats.load(tmp_path / "0", spec.stem)
    rec.num_rows += 1  # as if the episode had changed length
    rec.save(tmp_path / "0", spec.stem)
    fresh = Host(tmp_path, [0, 1, 2])
    fresh.generate_relative_action_stats(**SPEC_KW)
    assert EpisodeStats.load(tmp_path / "0", spec.stem).num_rows == LENGTHS[0] - 3


def test_errors(tmp_path):
    _write_episodes(tmp_path, LENGTHS)
    ds = Host(tmp_path, [0, 1, 2])
    with pytest.raises(ValueError, match="needs columns \\(0, 3\\)"):
        ds.generate_relative_action_stats(groups={"g": (0, 3)}, state_keys={"g": "observation/state/q"}, horizon=2)
    with pytest.raises(ValueError, match="valid keys"):
        ds.generate_relative_action_stats(groups={"g": (0, 2)}, state_keys={"g": "observation/state/nope"}, horizon=2)
    with pytest.raises(ValueError, match="fewer than horizon=50"):
        ds.generate_relative_action_stats(groups={"g": (0, 2)}, state_keys={"g": "observation/state/q"}, horizon=50)
    with pytest.raises(ValueError, match="fewer than|No chunks"):
        ds.generate_relative_action_stats(exact=True, groups={"g": (0, 2)}, state_keys={"g": "observation/state/q"}, horizon=50)


def test_state_dimension_must_match_the_group(tmp_path):
    _write_episodes(tmp_path, LENGTHS, dim=2, state_dim=3)
    with pytest.raises(ValueError, match="expected \\(rows, 2\\)"):
        Host(tmp_path, [0, 1, 2]).generate_relative_action_stats(**SPEC_KW)


def test_non_finite_names_the_episode(tmp_path):
    from tensordict import TensorDict

    _write_episodes(tmp_path, LENGTHS)
    td = TensorDict.load_memmap(str(tmp_path / "1"))
    td["action"][2, 0] = float("nan")
    with pytest.raises(ValueError, match="episode 1"):
        Host(tmp_path, [0, 1, 2]).generate_relative_action_stats(**SPEC_KW)


def test_compute_relative_episode_stats_none_for_short_episode(tmp_path):
    from tensordict import TensorDict

    _write_episodes(tmp_path, (3,))
    spec = build_specs(SPEC_KW["groups"], SPEC_KW["state_keys"], 4)[0]
    assert compute_relative_episode_stats(TensorDict.load_memmap(str(tmp_path / "0")), spec) is None


def test_gr00t_export_keeps_the_horizon_axis(tmp_path):
    _write_episodes(tmp_path, LENGTHS)
    ds = Host(tmp_path, [0, 1, 2])
    rel = ds.generate_relative_action_stats(**SPEC_KW)
    abs_stats = {
        "action": {k: [0.0, 0.0] for k in STATS},
        "observation/state/q": {k: [0.0, 0.0] for k in STATS},
    }
    g = to_gr00t_statistics(
        abs_stats, "new_embodiment", action_groups={"g": (0, 2)}, relative_action=rel
    )["new_embodiment"]
    assert set(g["relative_action"]["g"]) == set(STATS)
    assert np.asarray(g["relative_action"]["g"]["q99"]).shape == (4, 2)


# ---------------------------------------------------------------------------
# through a real dataset class
# ---------------------------------------------------------------------------

def test_lerobot_video_dataset_end_to_end(tmp_path):
    pytest.importorskip("torchrl")
    pytest.importorskip("imageio")
    pytest.importorskip("pandas")
    from robotdataset.lerobot_video_dataset import LeRobotVideoDataset
    from test_lerobot_video_dataset import make_dataset

    tree = make_dataset(tmp_path / "ds", episode_lengths=(6, 4, 5, 7, 3))
    ds = LeRobotVideoDataset(local_path=str(tree), episodes=[0, 1, 3], root=str(tmp_path / "cache"), batch_size=2)
    # action[:, 0] = 0.1 * t, observation.state.gripper = t  ->  chunk[h] = 0.1 * (t + h) - t
    kw = dict(groups={"g": (0, 1)}, state_keys={"g": "observation/state/gripper"}, horizon=3)
    got = ds.generate_relative_action_stats(**kw)["g"]
    chunks = np.asarray([[0.1 * (t + h) - t for h in range(3)] for n in (6, 4, 7) for t in range(n - 2)])
    _assert_stats(got, chunks, 3, 1, quantile_atol=1e-6)
    with pytest.raises(ValueError, match="expected \\(rows, 2\\)"):
        ds.generate_relative_action_stats(
            groups={"g": (0, 2)}, state_keys={"g": "observation/state/gripper"}, horizon=3
        )
    ds.sample()  # the dataset still samples
