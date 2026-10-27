"""DatasetInfo and Gr00tAdapter on a real OXEDataset over faked TFDS (needs the oxe extra, offline)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="oxe extras not installed")
pytest.importorskip("yaml")

import robotdataset.oxe_dataset as oxe  # noqa: E402
import test_oxe_dataset as oxe_t  # noqa: E402
from robotdataset.adapters.gr00t import Gr00tAdapter  # noqa: E402
from robotdataset.info import DatasetInfo  # noqa: E402

TAG = "new_embodiment"


@pytest.fixture()
def oxe_info(monkeypatch, tmp_path):
    oxe_t._patch(monkeypatch)
    kwargs = {"dataset_name": "droid", "root": str(tmp_path / "cache")}
    info = DatasetInfo.from_loader("OXEDataset", kwargs, episodes=[0])
    return info, kwargs


def _edited(info):
    pick = {
        "observation/image": dict(use=True, name="cam"),
        "observation/state": dict(use=True, split=(("xyz", (0, 3)), ("g", (3, 4)))),
        "action": dict(use=True, split=(("a_pos", (0, 1)), ("a_g", (1, 2)))),
        "language_instruction": dict(use=True),
    }
    mods = tuple(replace(m, **pick[m.path]) if m.path in pick else m for m in info.modalities)
    return replace(info, modalities=mods, extra={"gr00t": {"action_horizon": 3}})


def test_listing_of_an_oxe_loader(oxe_info):
    info, _ = oxe_info
    by = {m.path: m for m in info.modalities}
    assert by["observation/image"].kind == "image" and by["observation/image"].shape == (8, 8, 3)
    assert by["observation/state"].shape == (4,) and by["observation/state"].is_signal
    assert by["action"].kind == "action" and by["action"].width == 2
    assert by["language_instruction"].is_string  # a text leaf: its dtype is not a float
    assert not [p for p in by if p.startswith(("next/", "collector/")) or p in ("done", "terminated", "reward")]


def test_adapter_builds_the_oxe_loader_from_the_file(oxe_info):
    info, kwargs = oxe_info
    adapter = Gr00tAdapter.from_info(_edited(info))
    ds = adapter.build_dataset(episodes=[0, 1, 2], batch_size=4)  # loader, kwargs and windows come from the info
    assert type(ds).__name__ == "OXEDataset" and ds.num_episodes == 3
    assert adapter.image_paths >= {"observation/image"}  # OXE's image_keys is a list of strings
    b = ds.sample()
    assert tuple(b["action"].shape) == (4, 3, 2) and tuple(b["observation/state"].shape) == (4, 1, 4)
    assert tuple(b["observation/image"].shape) == (4, 1, 3, 8, 8)


def test_oxe_steps_and_statistics(oxe_info):
    pytest.importorskip("gr00t.data.types")
    info, _ = oxe_info
    adapter = Gr00tAdapter.from_info(_edited(info))
    ds = adapter.build_dataset(episodes=[0, 1, 2, 3, 4], batch_size=4)
    steps = adapter.to_steps(ds.sample(), TAG)
    s = steps[0]
    assert s.text == "pick up the block"  # the string leaf of the loader (load_str_fields=True)
    assert s.images["cam"][0].shape == (8, 8, 3)
    assert s.states["xyz"].shape == (1, 3) and s.states["g"].shape == (1, 1)
    assert s.actions["a_pos"].shape == (3, 1) and s.actions["a_g"].shape == (3, 1)
    stats = adapter.statistics(ds, TAG, exact=True)[TAG]
    assert list(stats["state"]) == ["xyz", "g"] and list(stats["action"]) == ["a_pos", "a_g"]
    assert "relative_action" not in stats  # nothing RELATIVE selected
    assert len(stats["state"]["xyz"]["mean"]) == 3
