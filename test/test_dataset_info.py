"""DatasetInfo: the generic modality listing, its YAML hand-off and its statistics (offline)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torchrl")
pytest.importorskip("datasets", reason="hf extras not installed")
pytest.importorskip("yaml")

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import robotdataset.table30v2_dataset as t30  # noqa: E402
from robotdataset.info import DatasetInfo, ModalityInfo, build_loader, main  # noqa: E402


# --------------------------------------------------------------------- Table30v2 (real loader, fake HF layer)


class _FakeHF:
    def __init__(self, rows):
        self._rows = list(rows)
        self.features = {k: None for k in rows[0]}

    def filter(self, fn):
        return _FakeHF([r for r in self._rows if fn(r)])

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        return [r[key] for r in self._rows] if isinstance(key, str) else self._rows[key]

    def __len__(self):
        return len(self._rows)


def _rows(n_eps=3, n_steps=12, seed=0):
    rng = np.random.default_rng(seed)
    f32 = np.float32
    rows = []
    for e in range(n_eps):
        for f in range(n_steps):
            rows.append({
                "episode_index": e,
                "frame_index": f,
                "observation.image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
                "observation.state": rng.normal(size=8).astype(f32),
                "observation.const": np.zeros(3, dtype=f32),
                "action": rng.normal(size=7).astype(f32),
                "next.reward": f32(0.0),
                "next.done": bool(f == n_steps - 1),
                "next.terminated": bool(f == n_steps - 1),
                "timestamp": f * 0.1,
                "task_index": 0,
            })
    return rows


@pytest.fixture()
def table30(monkeypatch, tmp_path):
    monkeypatch.setattr(t30, "load_hf_dataset", lambda *a, **k: _FakeHF(_rows()))
    kwargs = {"root": str(tmp_path / "cache")}
    ds = build_loader("Table30v2Dataset", kwargs, episodes=[0, 1, 2], batch_size=4)
    return ds, kwargs


def test_real_loader_listing(table30):
    ds, kwargs = table30
    info = DatasetInfo.from_dataset(ds, loader="Table30v2Dataset", kwargs=kwargs)
    assert info.loader == "Table30v2Dataset" and info.episodes_loaded == 3 and info.steps_loaded == 36
    by = {m.path: m for m in info.modalities}
    assert by["observation/image"].kind == "image" and by["observation/image"].shape == (8, 8, 3)  # HWC as stored
    assert by["observation/state"].shape == (8,) and by["observation/state"].is_signal
    assert by["action"].kind == "action" and by["action"].width == 7
    # TED bookkeeping is hidden
    assert not [p for p in by if p.startswith(("next/", "collector/")) or p in ("done", "terminated")]
    assert not any(m.use for m in info.modalities)  # nothing is selected until you say so
    assert len({m.name for m in info.modalities}) == len(info.modalities)


def test_statistics_stay_in_memory_and_show_in_the_table(table30):
    ds, kwargs = table30
    info = DatasetInfo.from_dataset(ds, kwargs=kwargs, stats=True, exact=True)
    assert info["observation/state"].stats["count"] == 36
    assert info["observation/image"].stats is None  # images carry no statistics
    table = info.table()
    assert "std min..max" in table and "3/3" in table  # the constant leaf is flagged
    assert "mean" not in info.to_yaml() and "q01" not in info.to_yaml()  # never written to the file


def test_cli_writes_a_loadable_file(table30, tmp_path, capsys):
    _, kwargs = table30
    out = tmp_path / "info.yaml"
    assert main(["--loader", "Table30v2Dataset", "--kwargs", json.dumps(kwargs), "--episodes", "0:2", "-o", str(out)]) == 0
    info = DatasetInfo.from_yaml(str(out))
    assert info.loader == "Table30v2Dataset" and info.kwargs == kwargs and info.episodes_loaded == 2
    assert "observation/state" in capsys.readouterr().out


def test_build_loader_rejects_unknown_and_reserved():
    with pytest.raises(ValueError, match="is not one of"):
        build_loader("OXEJAXDataset", {})
    with pytest.raises(ValueError, match="must not set"):
        build_loader("Table30v2Dataset", {"batch_size": 3})


# --------------------------------------------------------------------- loaders with other conventions (duck-typed)


class _Fake:
    """A loader double with a given listing and batch; implements what DatasetInfo uses."""

    def __init__(self, specs, image_keys, leaves, fps=None):
        self._specs, self.image_keys, self._leaves = specs, image_keys, leaves
        if fps is not None:
            self.fps = fps
        self.num_episodes = 2

    def __len__(self):
        return 20

    def get_modalities(self):
        return self._specs

    def sample(self):
        td = TensorDict({k: v for k, v in self._leaves.items() if torch.is_tensor(v)}, batch_size=[2])
        out = td.flatten_keys("/")
        for k, v in self._leaves.items():
            if not torch.is_tensor(v):
                out[k] = v
        return out

    def generate_stats(self, keys=None, exact=False):
        return {k: {"mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0], "q01": [0.0], "q99": [1.0], "count": 20} for k in keys}


def _oxe_like():
    """viola-like: dict-of-leaves action, scalar leaf, -1 dims from TFDS metadata, strings and an embedding."""
    spec = lambda kind, dtype, shape: {"kind": kind, "dtype": dtype, "shape": shape, "source": "metadata"}  # noqa: E731
    specs = {
        "observation/agentview_rgb": spec("image", "uint8", (224, 224, 3)),
        "observation/ee_states": spec("state", "float32", (16,)),
        "observation/natural_language_instruction": spec("text", "string", ()),
        "observation/natural_language_embedding": spec("text", "float32", (512,)),
        "action/world_vector": spec("action", "float32", (3,)),
        "action/gripper_closedness_action": spec("action", "float32", ()),
        "action/variable": spec("action", "float32", (-1,)),
        "done": spec("generic", "bool", (1,)),
        "next/observation/ee_states": spec("state", "float32", (16,)),
        "observation/not_in_storage": spec("state", "float32", (2,)),
    }
    leaves = {
        "observation/agentview_rgb": torch.zeros(2, 1, 3, 224, 224, dtype=torch.uint8),
        "observation/ee_states": torch.zeros(2, 1, 16),
        "observation/natural_language_instruction": ["a", "b"],
        "observation/natural_language_embedding": torch.zeros(2, 1, 512),
        "action/world_vector": torch.zeros(2, 1, 3),
        "action/gripper_closedness_action": torch.zeros(2, 1),
        "action/variable": torch.zeros(2, 1, 5),
        "done": torch.zeros(2, 1, dtype=torch.bool),
    }
    return _Fake(specs, ["observation/agentview_rgb"], leaves)


def test_oxe_like_listing_is_normalised():
    info = DatasetInfo.from_dataset(_oxe_like(), loader="OXEDataset")
    by = {m.path: m for m in info.modalities}
    assert "observation/not_in_storage" not in by and "done" not in by and "next/observation/ee_states" not in by
    text, emb = by["observation/natural_language_instruction"], by["observation/natural_language_embedding"]
    assert text.is_string and text.effective_role == "language"
    assert not emb.is_string and emb.is_signal  # a float embedding is data, not an instruction
    g = by["action/gripper_closedness_action"]
    assert g.shape == () and g.width == 1 and g.is_signal  # scalar leaf = one column
    assert by["action/variable"].shape == (5,)  # metadata says (-1,): the sampled batch gives the width
    assert info["observation/agentview_rgb"].effective_role == "video"
    assert len({m.name for m in info.modalities}) == len(info.modalities)


def test_lerobot_like_tuple_keys_and_fps():
    spec = lambda kind, dtype, shape: {"kind": kind, "dtype": dtype, "shape": str(shape)}  # noqa: E731
    specs = {
        "observation/images/primary": spec("image", "uint8", (256, 256, 3)),
        "observation/state/all": spec("state", "float32", (20,)),
        "observation/state/cartesian": spec("state", "float32", (6,)),
        "action": spec("action", "float32", (7,)),
        "observation/language_instruction": {"kind": "text", "dtype": None, "shape": None},
    }
    leaves = {
        "observation/images/primary": torch.zeros(2, 1, 3, 256, 256, dtype=torch.uint8),
        "observation/state/all": torch.zeros(2, 1, 20),
        "observation/state/cartesian": torch.zeros(2, 1, 6),
        "action": torch.zeros(2, 1, 7),
        "observation/language_instruction": ["x", "y"],
    }
    ds = _Fake(specs, frozenset({("observation", "images", "primary")}), leaves, fps=15)
    info = DatasetInfo.from_dataset(ds)
    assert info.fps == 15.0
    assert info["observation/language_instruction"].is_string  # dtype None: a Python string per step
    assert info["observation/state/all"].shape == (20,) and info["action"].shape == (7,)
    names = {m.path: m.name for m in info.modalities}
    assert names["observation/state/all"] == "all" and names["observation/state/cartesian"] == "cartesian"


def test_colliding_names_extend_to_a_unique_suffix():
    spec = {"kind": "state", "dtype": "float32", "shape": (2,)}
    ds = _Fake({"observation/left/pose": dict(spec), "observation/right/pose": dict(spec), "action": {"kind": "action", "dtype": "float32", "shape": (2,)}},
               [], {"observation/left/pose": torch.zeros(2, 1, 2), "observation/right/pose": torch.zeros(2, 1, 2), "action": torch.zeros(2, 1, 2)})
    names = {m.path: m.name for m in DatasetInfo.from_dataset(ds).modalities}
    assert names["observation/left/pose"] == "left_pose" and names["observation/right/pose"] == "right_pose" and names["action"] == "action"


# --------------------------------------------------------------------- the edited-YAML hand-off


def test_yaml_round_trip_keeps_edits_order_and_extra_blocks(tmp_path):
    info = DatasetInfo.from_dataset(_oxe_like(), loader="OXEDataset", kwargs={"dataset_name": "viola", "root": "/x"})
    text = info.to_yaml()
    assert "kwargs: {dataset_name: viola, root: /x}" in text
    data = __import__("yaml").safe_load(text)
    mods = data["modalities"]
    # edit: pick three modalities, reorder, rename, force a role, split a vector
    keep = {m["path"]: m for m in mods}
    order = ["action/world_vector", "observation/agentview_rgb", "observation/ee_states"]
    edited = [dict(keep[p], use=True) for p in order] + [m for m in mods if m["path"] not in order]
    edited[0]["name"] = "eef_pos"
    edited[2]["split"] = [{"name": "rot", "columns": [0, 9]}, {"name": "pos", "columns": [9, 16]}]
    edited[2]["role"] = "state"
    data["modalities"] = edited
    data["gr00t"] = {"action_horizon": 8}
    path = tmp_path / "edited.yaml"
    path.write_text(__import__("yaml").safe_dump(data, sort_keys=False))
    back = DatasetInfo.from_yaml(str(path))
    assert [m.path for m in back.selected] == order  # file order is the model order
    assert back.selected[0].name == "eef_pos" and back.selected[2].split == (("rot", (0, 9)), ("pos", (9, 16)))
    assert back.extra == {"gr00t": {"action_horizon": 8}}
    again = DatasetInfo.from_dict(__import__("yaml").safe_load(back.to_yaml()))  # and it survives a second pass
    assert again.modalities == back.modalities and again.extra == back.extra


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda d: d["modalities"][0].update(usee=True), "unknown field"),
        (lambda d: d["modalities"][0].update(role="video2"), "is not one of"),
        (lambda d: d["modalities"][1].update(path=d["modalities"][0]["path"]), "duplicate"),
        (lambda d: d["modalities"][0].update(split=[{"name": "a", "columns": [3, 3]}]), "start < end"),
        (lambda d: d["dataset"].pop("loader"), "loader is missing"),
        (lambda d: d.pop("modalities"), "missing top-level"),
    ],
)
def test_bad_edits_are_reported(edit, message):
    data = DatasetInfo.from_dataset(_oxe_like(), loader="OXEDataset").to_dict()
    edit(data)
    with pytest.raises(ValueError, match=message):
        DatasetInfo.from_dict(data)


def test_check_against_flags_drift_but_tolerates_unknown_dims():
    info = DatasetInfo.from_dataset(_oxe_like(), loader="OXEDataset")
    info.check_against(DatasetInfo.from_dataset(_oxe_like(), loader="OXEDataset"))  # identical: fine
    drifted = DatasetInfo(
        loader="OXEDataset",
        modalities=tuple(
            ModalityInfo(m.path, m.kind, "float64" if m.path == "observation/ee_states" else m.dtype,
                         (17,) if m.path == "action/world_vector" else m.shape, name=m.name)
            for m in info.modalities if m.path != "observation/agentview_rgb"
        ),
    )
    with pytest.raises(ValueError) as err:
        info.check_against(drifted)
    msg = str(err.value)
    assert "observation/agentview_rgb: not in the loader" in msg and "dtype float32 in the file, float64" in msg
    assert "shape [3] in the file, [17]" in msg and "action/variable:" not in msg
    # a file that still records an unknown (-1) dimension is not reported as drift
    loose = DatasetInfo(loader="OXEDataset", modalities=(ModalityInfo("action/variable", "action", "float32", (-1,)),))
    loose.check_against(info)


def test_parse_episodes_forms():
    from robotdataset.info import parse_episodes

    assert parse_episodes(None) is None
    assert parse_episodes(3) == [3]
    assert parse_episodes("0:4") == [0, 1, 2, 3]
    assert parse_episodes("0:2, 7") == [0, 1, 7]
    assert parse_episodes(["0:2", 5, "8:10"]) == [0, 1, 5, 8, 9]
    assert parse_episodes([]) == []
    with pytest.raises(TypeError):
        parse_episodes(True)


def test_string_leaves_the_loader_yields_but_does_not_describe_are_listed():
    from tensordict import NonTensorStack

    class Loader:
        num_episodes, fps = 1, 10.0

        def __len__(self):
            return 4

        def get_modalities(self):
            return {"observation/state": {"kind": "state", "dtype": "torch.float32", "shape": (3,)}}

        def sample(self):
            b = TensorDict({"observation/state": torch.zeros(2, 1, 3), "next/observation/state": torch.zeros(2, 1, 3)}, batch_size=[2])
            for key in ("observation/language_instruction", "next/observation/language_instruction"):
                b[key] = NonTensorStack("pick the block.", "pick the block.")
            return b

    info = DatasetInfo.from_dataset(Loader(), loader="LeRobotVideoDataset")
    assert [m.path for m in info.modalities] == ["observation/state", "observation/language_instruction"]
    text = info["observation/language_instruction"]
    assert text.kind == "text" and text.dtype == "string" and text.is_string and text.shape == ()
