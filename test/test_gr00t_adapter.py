"""Gr00tAdapter on a real Table30v2Dataset over a faked HuggingFace layer (offline)."""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torchrl")
pytest.importorskip("datasets", reason="hf extras not installed")
pytest.importorskip("yaml")

import yaml  # noqa: E402

import robotdataset.table30v2_dataset as t30  # noqa: E402
from robotdataset.adapters.gr00t import Gr00tAdapter  # noqa: E402
from robotdataset.info import DatasetInfo, ModalityInfo  # noqa: E402

TAG = "new_embodiment"

# ----------------------------------------------------------------------------- fake HF layer


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


def _rows(n_eps=4, n_steps=30, seed=0):
    rng = np.random.default_rng(seed)
    f32 = np.float32
    rows = []
    for e in range(n_eps):
        for f in range(n_steps):
            state = rng.normal(size=8).astype(f32)
            state[6] = f32(e * 100 + f)  # (episode, frame) readable through the sliced column
            rows.append({
                "episode_index": e,
                "frame_index": f,
                "observation.image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
                "observation.wrist": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
                "observation.state": state,
                "observation.pose": rng.normal(scale=0.1, size=6).astype(f32),
                "observation.joint": rng.normal(size=7).astype(f32),
                "observation.const": np.zeros(2, dtype=f32),
                "action.pose": rng.normal(scale=0.1, size=6).astype(f32),
                "action.joint": rng.normal(size=7).astype(f32),
                "action.gripper": np.array([e * 100 + f], dtype=f32),
                "next.reward": f32(0.0),
                "next.done": bool(f == n_steps - 1),
                "next.terminated": bool(f == n_steps - 1),
                "timestamp": f * 0.1,
                "task_index": 0,
            })
    return rows


GR00T_BLOCK = {
    "action_horizon": 4,
    "language": {"default": "stack the blocks."},
    "action": {
        "eef": {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROTVEC", "state_key": "pose"},
        "arm": {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "arm"},
    },
}

# path -> (use, name, role, split) for the edited file; order = the order below
PICKS = [
    ("observation/image", "front", None, None),
    ("observation/wrist", "wrist", "video", None),  # robotdataset's name heuristic does not flag this leaf as an image
    ("observation/pose", "pose", None, None),
    ("observation/joint", "arm", None, None),
    ("observation/state", None, None, (("gripper_state", (6, 7)),)),
    ("action/pose", "eef", None, None),
    ("action/joint", "arm", None, None),
    ("action/gripper", "gripper", None, None),
]


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(t30, "load_hf_dataset", lambda *a, **k: _FakeHF(_rows()))
    kwargs = {"root": str(tmp_path / "cache")}
    info = DatasetInfo.from_loader("Table30v2Dataset", kwargs, episodes=[0])
    return info, kwargs


def edited(info, block=GR00T_BLOCK, picks=PICKS):
    """The listing as a user would edit it: selected modalities first, in order, with names/roles/splits."""
    chosen = {p: (name, role, split) for p, name, role, split in picks}
    first = []
    for path, _, _, _ in picks:
        m = info[path]
        name, role, split = chosen[path]
        first.append(replace(m, use=True, name=name or m.name, role=role, split=split))
    rest = [m for m in info.modalities if m.path not in chosen]
    return replace(info, modalities=tuple(first + rest), extra={"gr00t": copy.deepcopy(block)})


@pytest.fixture()
def adapter(setup):
    return Gr00tAdapter.from_info(edited(setup[0]))


# ----------------------------------------------------------------------------- selection and layout


def test_order_names_and_assembled_positions(adapter):
    g = adapter.groups
    assert [x.name for x in g["video"]] == ["front", "wrist"]
    assert [x.name for x in g["state"]] == ["pose", "arm", "gripper_state"]
    assert [x.name for x in g["action"]] == ["eef", "arm", "gripper"]
    im = adapter.index_map()
    assert [(e["group"], e["assembled"]) for e in im["state"]] == [("pose", [0, 6]), ("arm", [6, 13]), ("gripper_state", [13, 14])]
    assert [(e["group"], e["assembled"]) for e in im["action"]] == [("eef", [0, 6]), ("arm", [6, 13]), ("gripper", [13, 14])]
    assert im["state"][2]["columns"] == [6, 7] and im["state"][2]["path"] == "observation/state"


def test_windows_are_steps_per_leaf(adapter):
    dt = adapter.delta_timestamps()
    assert dt["action/pose"] == [0.0, 1.0, 2.0, 3.0] and dt["observation/pose"] == [0.0]
    assert "observation/const" not in dt  # unselected leaves are not touched


def test_modality_configs_for_gr00t(adapter):
    pytest.importorskip("gr00t.data.types")
    cfgs = adapter.modality_configs()
    assert cfgs["video"].modality_keys == ["front", "wrist"] and cfgs["action"].delta_indices == [0, 1, 2, 3]
    assert [c.rep.name for c in cfgs["action"].action_configs] == ["RELATIVE", "RELATIVE", "ABSOLUTE"]
    assert cfgs["action"].action_configs[0].state_key == "pose"
    assert cfgs["language"].modality_keys == ["annotation.human.task_description"]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda i, b: [replace(m, use=False) if m.path in ("observation/image", "observation/wrist") else m for m in i.modalities], "no video modality selected"),
        (lambda i, b: [replace(m, role="language") if m.path == "observation/pose" else m for m in i.modalities], "language needs a text leaf"),
        (lambda i, b: [replace(m, role="video") if m.path == "action/pose" else m for m in i.modalities], "not an image"),
        (lambda i, b: [replace(m, name="front") if m.path == "observation/wrist" else m for m in i.modalities], "duplicate group name"),
        (lambda i, b: [replace(m, split=(("a", (0, 99)),)) if m.path == "observation/pose" else m for m in i.modalities], "exceed"),
        (lambda i, b: [replace(m, split=(("a", (0, 2)),)) if m.path == "observation/image" else m for m in i.modalities], "cannot be split"),
        (lambda i, b: b["action"]["eef"].update(state_key="nope"), "must name a state group"),
        (lambda i, b: b["action"].update(ghost={"rep": "ABSOLUTE"}), "no selected action group named"),
        (lambda i, b: b["action"]["eef"].update(rep="RELATVE"), "is not one of"),
        (lambda i, b: b.update(horizon=3), "unknown field"),
        (lambda i, b: b["language"].update(default=None), "needs an instruction"),
    ],
)
def test_selection_errors_say_what_to_fix(setup, edit, message):
    info = edited(setup[0])
    block = copy.deepcopy(GR00T_BLOCK)
    mods = edit(info, block)
    if mods is not None:
        info = replace(info, modalities=tuple(mods))
    info = replace(info, extra={"gr00t": block})
    with pytest.raises(ValueError, match=message):
        Gr00tAdapter.from_info(info)


def test_conflicting_windows_on_one_leaf(adapter):
    from robotdataset.adapters.gr00t import Group

    # a state group on the same leaf as an action group: window [0] against 0..3
    adapter.groups["state"].append(Group("clash", "action/pose", "state"))
    with pytest.raises(ValueError, match="two different windows"):
        adapter.delta_timestamps()


# ----------------------------------------------------------------------------- loader, batches, steps


def test_build_dataset_binds_and_the_batch_has_the_selected_layout(adapter):
    ds = adapter.build_dataset(episodes=[0, 1, 2], batch_size=6)
    assert ds.num_episodes == 3 and "observation/image" in adapter.image_paths and "observation/wrist" not in adapter.image_paths
    b = ds.sample()
    assert tuple(b["action/pose"].shape) == (6, 4, 6) and tuple(b["observation/state"].shape) == (6, 1, 8)
    assert tuple(b["observation/image"].shape) == (6, 1, 3, 8, 8)  # flagged image: CHW from the loader
    assert tuple(b["observation/wrist"].shape) == (6, 1, 8, 8, 3)  # not flagged: stays HWC


def test_bind_reports_a_changed_loader(adapter, setup):
    ds = adapter.build_dataset(episodes=[0], batch_size=2)
    bad = replace(adapter.info, modalities=tuple(
        replace(m, shape=(99,)) if m.path == "observation/pose" else m for m in adapter.info.modalities))
    with pytest.raises(ValueError, match=r"observation/pose: shape \[99\] in the file, \[6\]"):
        Gr00tAdapter.from_info(bad).bind(ds)


def test_check_batch_lists_problems(adapter):
    ds = adapter.build_dataset(episodes=[0], batch_size=2)
    other = Gr00tAdapter.from_info(replace(adapter.info, extra={"gr00t": {**GR00T_BLOCK, "action_horizon": 7}}))
    other.image_paths = adapter.image_paths
    with pytest.raises(ValueError, match="expected ndim"):
        other.check_batch(ds.sample())  # sampled with horizon 4, adapter now asks for 7


def test_to_steps_slices_columns_and_layouts(adapter):
    pytest.importorskip("gr00t.data.types")
    ds = adapter.build_dataset(episodes=[0, 1, 2], batch_size=6)
    batch = ds.sample()
    steps = adapter.to_steps(batch, TAG)
    s = steps[0]
    assert len(steps) == 6 and s.text == "stack the blocks." and getattr(s.embodiment, "value", s.embodiment) == TAG
    assert s.images["front"][0].shape == (8, 8, 3) and s.images["front"][0].dtype == np.uint8
    assert s.states["gripper_state"].shape == (1, 1) and s.states["pose"].shape == (1, 6) and s.states["arm"].shape == (1, 7)
    assert s.actions["eef"].shape == (4, 6) and s.actions["gripper"].shape == (4, 1)
    np.testing.assert_array_equal(
        np.stack([x.states["gripper_state"][0, 0] for x in steps]), batch["observation/state"][:, 0, 6].numpy()
    )
    np.testing.assert_array_equal(steps[1].images["front"][0], batch["observation/image"][1, 0].permute(1, 2, 0).numpy())
    np.testing.assert_array_equal(steps[1].images["wrist"][0], batch["observation/wrist"][1, 0].numpy())
    g = steps[2].actions["gripper"][:, 0]
    assert np.all(np.diff(g) >= 0) and np.all(np.diff(g) <= 1)  # anchor, anchor+1, ... (clamped at the end)


def test_to_model_batch_runs_every_step_through_the_processor(adapter):
    gtypes = pytest.importorskip("gr00t.data.types")
    ds = adapter.build_dataset(episodes=[0, 1, 2], batch_size=5)
    seen = []

    class Processor:
        def __call__(self, messages):
            assert [m["type"] for m in messages] == [gtypes.MessageType.EPISODE_STEP.value]
            seen.append(messages[0]["content"])
            return {"n": len(seen)}

        def collator(self, features):
            return {"features": features}

    out = adapter.to_model_batch(ds.sample(), Processor(), TAG)
    assert [f["n"] for f in out["features"]] == [1, 2, 3, 4, 5]
    assert all(isinstance(x, gtypes.VLAStepData) and getattr(x.embodiment, "value", x.embodiment) == TAG for x in seen)


# ----------------------------------------------------------------------------- statistics


def test_statistics_come_from_the_loader_and_follow_the_selection(adapter):
    ds = adapter.build_dataset(episodes=[0, 1, 2], batch_size=6)
    stats = adapter.statistics(ds, TAG, exact=True)[TAG]
    assert set(stats) == {"state", "action", "relative_action"}
    assert list(stats["state"]) == ["pose", "arm", "gripper_state"] and list(stats["action"]) == ["eef", "arm", "gripper"]
    assert set(stats["relative_action"]) == {"eef", "arm"}
    full = ds.generate_stats(keys=["observation/state"], exact=True)["observation/state"]
    assert stats["state"]["gripper_state"]["mean"] == [full["mean"][6]]  # column 6 of the assembled leaf
    assert len(stats["action"]["eef"]["q01"]) == 6 and len(stats["state"]["arm"]["q99"]) == 7
    assert np.shape(stats["relative_action"]["arm"]["mean"]) == (4, 7)  # (horizon, D)


def test_configure_processor_hands_statistics_over(adapter):
    ds = adapter.build_dataset(episodes=[0, 1, 2], batch_size=6)
    seen = {}

    class Processor:
        def set_statistics(self, stats, override=False):
            seen.update(stats=stats, override=override)

    out = adapter.configure_processor(Processor(), ds, TAG)
    assert seen["override"] is True and seen["stats"] is out and TAG in out


def test_relative_reference_state_slice_is_refused(setup):
    info = edited(setup[0])
    mods = tuple(replace(m, split=(("pose", (0, 6)),)) if m.path == "observation/state" else m for m in info.modalities)
    # make the 'pose' reference a slice of the assembled state instead of its own leaf
    mods = tuple(replace(m, use=False) if m.path == "observation/pose" else m for m in mods)
    a = Gr00tAdapter.from_info(replace(info, modalities=mods))
    ds = a.build_dataset(episodes=[0, 1, 2], batch_size=4)
    with pytest.raises(NotImplementedError, match="whole TED leaf"):
        a.statistics(ds, TAG)


def test_zero_range_columns_warn(setup):
    info = edited(setup[0])
    mods = tuple(replace(m, use=True, name="const", role="state") if m.path == "observation/const" else m for m in info.modalities)
    a = Gr00tAdapter.from_info(replace(info, modalities=mods))
    ds = a.build_dataset(episodes=[0, 1, 2], batch_size=4)
    with pytest.warns(UserWarning, match="zero q01..q99 range"):
        a.statistics(ds, TAG)


# ----------------------------------------------------------------------------- the YAML hand-off


def test_defaults_block_cli_and_edited_file_round_trip(setup, tmp_path, monkeypatch, capsys):
    from robotdataset.info import main

    _, kwargs = setup
    out = tmp_path / "modality.yaml"
    assert main(["--loader", "Table30v2Dataset", "--kwargs", __import__("json").dumps(kwargs), "--episodes", "0:2",
                 "--adapter", "gr00t", "-o", str(out)]) == 0
    text = out.read_text()
    assert "# Isaac-GR00T settings" in text and "gr00t:" in text and "action_horizon: 16" in text
    # an unedited file selects nothing, so the adapter refuses it and says what to do
    with pytest.raises(ValueError, match="no video modality selected"):
        Gr00tAdapter.from_yaml(str(out))
    # edit the file the way a user would, then load it
    data = yaml.safe_load(text)
    data["gr00t"]["action_horizon"] = 4
    data["gr00t"]["language"]["default"] = "stack the blocks."
    for m in data["modalities"]:
        m["use"] = m["path"] in ("observation/image", "observation/state", "action/pose")
    data["modalities"] = sorted(data["modalities"], key=lambda m: m["path"] != "observation/state")  # state first
    for m in data["modalities"]:
        if m["path"] == "observation/state":
            m["split"] = [{"name": "arm_state", "columns": [0, 6]}, {"name": "gripper_state", "columns": [6, 7]}]
        if m["path"] == "action/pose":
            m["name"] = "eef"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    a = Gr00tAdapter.from_yaml(str(out))
    assert [g.name for g in a.groups["state"]] == ["arm_state", "gripper_state"] and a.action_horizon == 4
    ds = a.build_dataset(episodes=[0, 1], batch_size=3)  # loader and kwargs come from the file
    assert ds.num_episodes == 2 and tuple(ds.sample()["action/pose"].shape) == (3, 4, 6)


def test_gr00t_export_functions_keep_their_old_import_path():
    from robotdataset.gr00t import GR00T_STAT_KEYS, check_against, to_gr00t_statistics

    assert GR00T_STAT_KEYS == ("mean", "std", "min", "max", "q01", "q99")
    assert callable(check_against) and callable(to_gr00t_statistics)


def test_generated_gr00t_block_carries_its_own_notes_and_reads_back(setup):
    _, kwargs = setup
    info = Gr00tAdapter.with_defaults(DatasetInfo.from_loader("Table30v2Dataset", kwargs, episodes=[0, 1]))
    text = info.to_yaml()  # no help text is passed: the registered adapter supplies it
    assert "# Isaac-GR00T settings (read by Gr00tAdapter).\ngr00t:\n  action_horizon: 16  # action delta_indices" in text
    assert "  video_delta: [0]    # " in text and "  state_delta: [0]    # " in text
    assert "\n  - 0" not in text  # lists stay on one line
    assert "  action: {}  # per action group" in text and "  #   eef:     {rep: RELATIVE" in text
    assert yaml.safe_load(text)["gr00t"] == Gr00tAdapter.DEFAULTS
    assert "# Isaac-GR00T settings" not in info.to_yaml(comments={"gr00t": ""})
    # once entries replace the {}, one aligned row per action group
    entries = {
        "pose": {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROTVEC", "state_key": "pose"},
        "gripper": {"rep": "ABSOLUTE", "type": "NON_EEF"},
    }
    filled = DatasetInfo.from_dict({**info.to_dict(), "gr00t": {**info.extra["gr00t"], "action": entries}})
    text = filled.to_yaml()
    rows = [ln for ln in text.splitlines() if ln.startswith(("    pose:", "    gripper:"))]
    assert len(rows) == 2 and rows[0].index("{rep") == rows[1].index("{rep")
    assert yaml.safe_load(text)["gr00t"]["action"] == entries
