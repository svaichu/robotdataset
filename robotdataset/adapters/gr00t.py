"""Isaac-GR00T adapter: dataset info and picks -> GR00T modality configs, steps and statistics.

Kept apart from :mod:`robotdataset.stats` and :mod:`robotdataset.info` so the core stays
GR00T-agnostic; the ``gr00t`` package itself is imported lazily.  GR00T's processor takes
``{embodiment_tag: {modality: {group: {stat: [floats]}}}}`` with ``modality`` in ``state``,
``action``, ``relative_action`` and ``stat`` in ``mean, std, min, max, q01, q99``.  Group names
must match the ``modality_keys`` of the GR00T ``ModalityConfig``; a group's vector length
defines the dimension used to split a predicted action back into groups.

Workflow::

    python -m robotdataset.info --loader LeRobotVideoDataset --kwargs '{"local_path": "..."}' \
        --adapter gr00t -o modality.yaml          # then edit: use / order / name / split / gr00t block

    adapter = Gr00tAdapter.from_yaml("modality.yaml")
    train_ds = adapter.build_dataset(episodes=range(46), batch_size=8)   # loader + windows + checks
    processor = Gr00tN1d7Processor(modality_configs={tag: adapter.modality_configs()}, ...)
    adapter.configure_processor(processor, train_ds, tag)                 # statistics, in memory
    steps = adapter.to_steps(train_ds.sample(), embodiment_tag)           # list of VLAStepData

The ``gr00t:`` block of the file::

    gr00t:
      action_horizon: 16                 # action delta_indices = 0 .. horizon-1
      video_delta: [0]                   # steps relative to the anchor
      state_delta: [0]
      language:
        default: null                    # instruction used when no text leaf is selected
        modality_key: annotation.human.task_description   # the key GR00T looks the text up under
      action:                            # per action group (omitted group = ABSOLUTE / NON_EEF / DEFAULT)
        eef: {rep: RELATIVE, type: EEF, format: XYZ_ROTVEC, state_key: cartesian}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from robotdataset.adapters import ModelAdapter
from robotdataset.info import DatasetInfo, ModalityInfo

GR00T_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
_STATE_PREFIX = "observation/state/"


def _slice(stat: Mapping[str, Any], start: int, end: int, dim: int, what: str) -> Dict[str, list]:
    if not (0 <= start < end <= dim):
        raise ValueError(
            f"{what}: range ({start}, {end}) is invalid for dimension {dim}; "
            "need 0 <= start < end <= dim"
        )
    return {k: list(stat[k][start:end]) for k in GR00T_STAT_KEYS}


def _whole(stat: Mapping[str, Any]) -> Dict[str, list]:
    return {k: list(stat[k]) for k in GR00T_STAT_KEYS}


def _require(stats: Mapping[str, Any], key: str, what: str) -> Mapping[str, Any]:
    if key not in stats:
        raise ValueError(f"{what}: key {key!r} is missing from stats; available: {sorted(stats)}")
    missing = [k for k in GR00T_STAT_KEYS if k not in stats[key]]
    if missing:
        raise ValueError(f"{what}: stats for {key!r} lack {missing}")
    return stats[key]


def _ranges(meta: Mapping[str, Mapping[str, Any]]) -> Dict[str, Tuple[int, int]]:
    return {g: (int(r["start"]), int(r["end"])) for g, r in meta.items()}


def to_gr00t_statistics(
    stats: Mapping[str, Mapping[str, Any]],
    embodiment_tag: str,
    state_groups: Optional[Mapping[str, str]] = None,
    action_groups: Optional[Mapping[str, Tuple[int, int]]] = None,
    action_key: str = "action",
    modality_meta: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    relative_action: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, list]]]]:
    """Convert ``ds.generate_stats()`` output to GR00T's statistics layout.

    Args:
        stats: Output of ``generate_stats()``.
        embodiment_tag: Top-level key of the result.
        state_groups: ``{group_name: stats_key}``.  ``None`` uses every leaf
            under ``observation/state/`` (group name = the rest of the path).
            When both ``observation/state/all`` and per-slice leaves exist,
            ``all`` is dropped because it duplicates the slices.
        action_groups: ``{group_name: (start, end)}`` slicing the action vector.
        action_key: Stats key holding the concatenated action.
        modality_meta: Same structure as GR00T's ``meta/modality.json``
            (``{"action": {group: {"start": a, "end": b}}}``), as an
            alternative to ``action_groups``.  Its ``"state"`` entries slice the
            key given by their ``"original_key"`` (default
            ``observation/state/all``) when ``state_groups`` is ``None``.
        relative_action: ``{group_name: stat dict}`` computed elsewhere;
            emitted under ``relative_action``.

    Returns:
        ``{embodiment_tag: {"state": {...}, "action": {...}}}`` with exactly the
        six stat keys per group (``count`` is dropped).

    Raises:
        ValueError: Invalid ranges, or a requested group/key missing from ``stats``.

    Note:
        ``processor.set_statistics(g, override=False)`` keeps the existing
        statistics of an embodiment tag that is already present and discards the
        new ones with only a log warning.  Pass ``override=True`` when
        finetuning with new statistics.

    Example::

        ds = LeRobotVideoDataset(repo_id="...", episodes=train_ids, batch_size=8)
        g = to_gr00t_statistics(
            ds.generate_stats(),
            embodiment_tag="new_embodiment",
            action_groups={"arm": (0, 7), "gripper": (7, 8)},
        )
        processor.set_statistics(g, override=True)
    """
    if action_groups is not None and modality_meta is not None and "action" in modality_meta:
        raise ValueError("Pass either action_groups or modality_meta['action'], not both")
    meta = modality_meta or {}
    if action_groups is None:
        if "action" not in meta:
            raise ValueError("Specify action_groups or modality_meta['action']")
        action_groups = _ranges(meta["action"])

    # ---- state ----
    state: Dict[str, Dict[str, list]] = {}
    if state_groups is not None:
        for group, key in state_groups.items():
            state[group] = _whole(_require(stats, key, f"state group {group!r}"))
    elif "state" in meta:
        for group, r in meta["state"].items():
            key = str(r.get("original_key", "observation/state/all")).replace(".", "/")
            src = _require(stats, key, f"state group {group!r}")
            state[group] = _slice(
                src, int(r["start"]), int(r["end"]), len(src["mean"]), f"state group {group!r}"
            )
    else:
        keys = sorted(k for k in stats if k.startswith(_STATE_PREFIX))
        if len(keys) > 1:
            keys = [k for k in keys if k != _STATE_PREFIX + "all"]
        for key in keys:
            state[key[len(_STATE_PREFIX):]] = _whole(_require(stats, key, f"state key {key!r}"))
    if not state:
        raise ValueError(
            "No state groups: stats has no 'observation/state/*' keys and no "
            "state_groups / modality_meta['state'] was given"
        )

    # ---- action ----
    src = _require(stats, action_key, "action")
    dim = len(src["mean"])
    action = {
        g: _slice(src, int(a), int(b), dim, f"action group {g!r}")
        for g, (a, b) in action_groups.items()
    }

    out: Dict[str, Dict[str, Dict[str, list]]] = {"state": state, "action": action}
    if relative_action is not None:
        out["relative_action"] = {
            g: _whole(_require(relative_action, g, f"relative_action group {g!r}"))
            for g in relative_action
        }
    return {embodiment_tag: out}


def check_against(
    gr00t_stats: Mapping[str, Mapping[str, Mapping[str, Any]]],
    modality_configs: Mapping[str, Any],
    embodiment_tag: Optional[str] = None,
) -> None:
    """Compare group names to GR00T modality configs; raises ``ValueError`` on differences.

    Args:
        gr00t_stats: Output of :func:`to_gr00t_statistics`.
        modality_configs: ``{modality: config}`` where ``config`` is a list of
            keys, a mapping with ``"modality_keys"``, or an object with a
            ``modality_keys`` attribute (GR00T's ``ModalityConfig``).
        embodiment_tag: Which tag to check; defaults to the only one present.
    """
    if embodiment_tag is None:
        if len(gr00t_stats) != 1:
            raise ValueError(f"Several embodiment tags {sorted(gr00t_stats)}; pass embodiment_tag")
        embodiment_tag = next(iter(gr00t_stats))
    mine = gr00t_stats[embodiment_tag]
    problems = []
    for modality, cfg in modality_configs.items():
        if modality not in ("state", "action"):
            continue
        if isinstance(cfg, Mapping):
            keys = cfg["modality_keys"]
        else:
            keys = getattr(cfg, "modality_keys", cfg)
        expected = [k.split(".", 1)[-1] for k in keys]
        have = list(mine.get(modality, {}))
        if sorted(expected) != sorted(have):
            problems.append(f"{modality}: config groups {sorted(expected)} != stats groups {sorted(have)}")
    if problems:
        raise ValueError("; ".join(problems))


# =============================================================================
# The adapter
# =============================================================================

REPS = ("ABSOLUTE", "RELATIVE", "DELTA")
TYPES = ("EEF", "NON_EEF")
FORMATS = ("DEFAULT", "XYZ_ROT6D", "XYZ_ROTVEC")
_BLOCK_FIELDS = {"action_horizon", "video_delta", "state_delta", "language", "action"}
_LANGUAGE_FIELDS = {"default", "modality_key"}
_ACTION_FIELDS = {"rep", "type", "format", "state_key"}
_RELATIVE_REPRESENTATION = {("EEF", "XYZ_ROTVEC"): "xyz_rotvec", ("NON_EEF", "DEFAULT"): "joint"}


def _paths(keys: Iterable[Any]) -> set:
    """``ds.image_keys`` as ``"/"`` strings (loaders return tuples or strings)."""
    return {"/".join(k) if isinstance(k, tuple) else str(k) for k in keys}


def _ints(value: Any, where: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{where}: expected a list of ints, got {value!r}")
    return tuple(value)


@dataclass(frozen=True)
class Group:
    """One GR00T group: a TED leaf, or columns ``[start:end)`` of it."""

    name: str
    path: str
    role: str  # video | state | action
    start: Optional[int] = None
    end: Optional[int] = None
    rep: str = "ABSOLUTE"
    type: str = "NON_EEF"
    format: str = "DEFAULT"
    state_key: Optional[str] = None

    @property
    def is_slice(self) -> bool:
        return self.start is not None

    def columns(self, width: int) -> Tuple[int, int]:
        return (self.start, self.end) if self.is_slice else (0, width)

    def take(self, x: Any) -> Any:
        return x[..., self.start : self.end] if self.is_slice else x


class Gr00tAdapter(ModelAdapter):
    """Dataset info -> GR00T: groups and their order, windows, configs, steps and statistics."""

    name = "gr00t"
    YAML_HELP = (
        "Isaac-GR00T settings (read by Gr00tAdapter).\n"
        "  action_horizon: action delta_indices are 0 .. horizon-1\n"
        "  video_delta / state_delta: step offsets relative to the anchor\n"
        "  language: default instruction when no text modality is selected; modality_key is the name GR00T looks it up under\n"
        "  action: per action group {rep: ABSOLUTE|RELATIVE|DELTA, type: EEF|NON_EEF, format: DEFAULT|XYZ_ROT6D|XYZ_ROTVEC,\n"
        "          state_key: <state group name, required for RELATIVE>}; a group left out is ABSOLUTE / NON_EEF / DEFAULT"
    )

    DEFAULTS: Dict[str, Any] = {
        "action_horizon": 16,
        "video_delta": [0],
        "state_delta": [0],
        "language": {"default": None, "modality_key": "annotation.human.task_description"},
        "action": {},
    }

    def __init__(self, info: DatasetInfo) -> None:
        super().__init__(info)
        block = self.block
        unknown = sorted(set(block) - _BLOCK_FIELDS)
        if unknown:
            raise ValueError(f"gr00t block: unknown field(s) {unknown}; allowed: {sorted(_BLOCK_FIELDS)}")
        self.action_horizon = int(block.get("action_horizon", self.DEFAULTS["action_horizon"]))
        if self.action_horizon < 1:
            raise ValueError(f"gr00t.action_horizon must be >= 1, got {self.action_horizon}")
        self.video_delta = _ints(block.get("video_delta", self.DEFAULTS["video_delta"]), "gr00t.video_delta")
        self.state_delta = _ints(block.get("state_delta", self.DEFAULTS["state_delta"]), "gr00t.state_delta")
        lang = {**self.DEFAULTS["language"], **(block.get("language") or {})}
        extra = sorted(set(lang) - _LANGUAGE_FIELDS)
        if extra:
            raise ValueError(f"gr00t.language: unknown field(s) {extra}; allowed: {sorted(_LANGUAGE_FIELDS)}")
        self.language_default: Optional[str] = lang["default"]
        self.language_modality_key: str = lang["modality_key"]
        self.image_paths: set = set()
        self.language_path: Optional[str] = None
        self.groups: Dict[str, List[Group]] = self._build_groups(block.get("action") or {})

    # ------------------------------------------------------------------ construction

    @classmethod
    def with_defaults(cls, info: DatasetInfo) -> DatasetInfo:
        from dataclasses import replace

        return replace(info, extra={**dict(info.extra), cls.name: {k: v for k, v in cls.DEFAULTS.items()}})

    def _build_groups(self, action_block: Mapping[str, Any]) -> Dict[str, List[Group]]:
        groups: Dict[str, List[Group]] = {"video": [], "state": [], "action": []}
        language_leaf: Optional[ModalityInfo] = None
        for m in self.info.selected:
            role = m.effective_role
            where = f"{m.path} (kind {m.kind})"
            if role is None:
                raise ValueError(f"{where}: no default role; set 'role: video | state | action | language' on it")
            if role == "language":
                if not m.is_string:
                    raise ValueError(f"{where}: language needs a text leaf (a string per step), not a float embedding")
                if language_leaf is not None:
                    raise ValueError(f"two text modalities selected ({language_leaf.path}, {m.path}); GR00T takes one instruction")
                language_leaf = m
                continue
            name = m.name or m.path.split("/")[-1]
            if role == "video":
                if not m.looks_like_image:
                    raise ValueError(f"{where}: selected as video but it is not an image (uint8 HWC); set role if the heuristic is wrong")
                if m.split:
                    raise ValueError(f"{where}: images cannot be split")
                groups["video"].append(Group(name, m.path, "video"))
                continue
            if not m.is_signal or m.width is None:
                raise ValueError(f"{where}: {role} needs a float vector leaf of known width, got dtype {m.dtype} shape {list(m.shape)}")
            pieces = [(n, c) for n, c in m.split] if m.split else [(name, None)]
            for n, cols in pieces:
                if cols is not None and cols[1] > m.width:
                    raise ValueError(f"{where}: split {n!r} columns {list(cols)} exceed the leaf's {m.width} columns")
                sem = {}
                if role == "action":
                    sem = dict(action_block.get(n) or {})
                    bad = sorted(set(sem) - _ACTION_FIELDS)
                    if bad:
                        raise ValueError(f"gr00t.action.{n}: unknown field(s) {bad}; allowed: {sorted(_ACTION_FIELDS)}")
                    for key, allowed in (("rep", REPS), ("type", TYPES), ("format", FORMATS)):
                        if key in sem and sem[key] not in allowed:
                            raise ValueError(f"gr00t.action.{n}.{key}: {sem[key]!r} is not one of {allowed}")
                groups[role].append(
                    Group(n, m.path, role, *(cols if cols else (None, None)), **sem)
                )
        self.language_path = language_leaf.path if language_leaf else None

        for role, gs in groups.items():
            names = [g.name for g in gs]
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise ValueError(f"{role}: duplicate group name(s) {dup}; rename them with 'name:' or 'split'")
            if not gs:
                raise ValueError(
                    f"no {role} modality selected: set 'use: true' on at least one "
                    f"{'image' if role == 'video' else role} modality (use roles to override a wrong kind)"
                )
        if self.language_path is None and self.language_default is None:
            raise ValueError("no text modality selected and gr00t.language.default is null: GR00T needs an instruction")
        state_names = {g.name for g in groups["state"]}
        for g in groups["action"]:
            if g.rep == "RELATIVE" and g.state_key not in state_names:
                raise ValueError(
                    f"action group {g.name!r} is RELATIVE: state_key {g.state_key!r} must name a state group (have {sorted(state_names)})"
                )
        unknown_actions = sorted(set(action_block) - {g.name for g in groups["action"]})
        if unknown_actions:
            raise ValueError(
                f"gr00t.action: no selected action group named {unknown_actions} (groups: {[g.name for g in groups['action']]})"
            )
        return groups

    # ------------------------------------------------------------------ layout

    def index_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Where each group sits in the assembled state / action vector (groups concatenated in order)."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for role in ("state", "action"):
            pos, entries = 0, []
            for g in self.groups[role]:
                a, b = g.columns(self.info[g.path].width)
                entries.append({"group": g.name, "path": g.path, "columns": [a, b], "assembled": [pos, pos + b - a]})
                pos += b - a
            out[role] = entries
        return out

    def delta_timestamps(self) -> Dict[str, List[float]]:
        """``{TED path: offsets}`` in steps; use with ``control_frequency=1.0`` (see ``build_dataset``)."""
        out: Dict[str, Tuple[int, ...]] = {}

        def add(path: str, deltas: Tuple[int, ...], role: str) -> None:
            if out.setdefault(path, deltas) != deltas:
                raise ValueError(
                    f"{path!r} is read at offsets {list(out[path])} and, as {role}, at {list(deltas)}; "
                    "one leaf cannot be sampled at two different windows"
                )

        for g in self.groups["video"]:
            add(g.path, self.video_delta, "video")
        for g in self.groups["state"]:
            add(g.path, self.state_delta, "state")
        for g in self.groups["action"]:
            add(g.path, tuple(range(self.action_horizon)), "action")
        return {k: [float(i) for i in v] for k, v in out.items()}

    def modality_configs(self) -> dict:
        """GR00T ``{modality: ModalityConfig}`` for ``Gr00tN1d7Processor``."""
        from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

        g = self.groups
        return {
            "video": ModalityConfig(delta_indices=list(self.video_delta), modality_keys=[x.name for x in g["video"]]),
            "state": ModalityConfig(delta_indices=list(self.state_delta), modality_keys=[x.name for x in g["state"]]),
            "action": ModalityConfig(
                delta_indices=list(range(self.action_horizon)),
                modality_keys=[x.name for x in g["action"]],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation[x.rep],
                        type=ActionType[x.type],
                        format=ActionFormat[x.format],
                        state_key=x.state_key,
                    )
                    for x in g["action"]
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[self.language_modality_key]),
        }

    # ------------------------------------------------------------------ checks

    def bind(self, ds: Any) -> None:
        """Validate against a built loader and remember its image layout.

        Checks the file against a fresh listing of the loader (paths, dtype, shape of the selected
        modalities) and a sampled batch against the selection (windows, shapes, column bounds).
        """
        fresh = DatasetInfo.from_dataset(ds, loader=self.info.loader)
        used = {g.path for gs in self.groups.values() for g in gs} | ({self.language_path} if self.language_path else set())
        self.info.check_against(fresh, only=used)
        self.image_paths = _paths(ds.image_keys)
        self.check_batch(ds.sample())

    def check_batch(self, batch: Any) -> None:
        """Raise one ``ValueError`` listing every way ``batch`` differs from the selection."""
        have = set(batch.keys())
        problems: List[str] = []

        def leaf(group: Group, ndims: Sequence[int], steps: int) -> Any:
            if group.path not in have:
                problems.append(f"{group.role}.{group.name}: {group.path!r} not in batch (have: {sorted(have)})")
                return None
            x = batch[group.path]
            if x.ndim not in ndims or x.shape[1] != steps:
                problems.append(
                    f"{group.role}.{group.name}: {group.path!r} has shape {tuple(x.shape)}; expected ndim in "
                    f"{list(ndims)} with {steps} step(s) on axis 1 (was the loader built by build_dataset?)"
                )
                return None
            return x

        for g in self.groups["video"]:
            x = leaf(g, (5,), len(self.video_delta))
            if x is not None:
                channels = x.shape[2] if g.path in self.image_paths else x.shape[-1]
                if channels not in (1, 3, 4):
                    problems.append(f"video.{g.name}: {g.path!r} {tuple(x.shape)} does not look like images")
        for role, steps in (("state", len(self.state_delta)), ("action", self.action_horizon)):
            for g in self.groups[role]:
                x = leaf(g, (2, 3), steps)
                if x is not None and g.is_slice and g.end > (x.shape[-1] if x.ndim == 3 else 1):
                    problems.append(f"{role}.{g.name}: columns {[g.start, g.end]} exceed the leaf's width")
        if self.language_path is not None and self.language_path not in have and self.language_default is None:
            problems.append(f"language: {self.language_path!r} not in batch and gr00t.language.default is null")
        if problems:
            raise ValueError("the selection does not match the sampled batch:\n  " + "\n  ".join(problems))

    # ------------------------------------------------------------------ batch -> VLAStepData

    def to_steps(self, batch: Any, embodiment_tag: Any) -> list:
        """One GR00T ``VLAStepData`` per sample: HWC uint8 images, ``(T, D)`` states, ``(horizon, D)`` actions.

        ``embodiment_tag`` is an ``EmbodimentTag`` or its name or value (``"NEW_EMBODIMENT"``, ``"new_embodiment"``).
        """
        import numpy as np
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import VLAStepData

        if isinstance(embodiment_tag, str):  # the processor reads ``embodiment.value``, so a string becomes the enum
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        n = batch.batch_size[0]

        def images(g: Group) -> Any:
            x = batch[g.path]
            if g.path in self.image_paths:  # robotdataset permutes flagged image leaves to CHW
                x = x.permute(0, 1, 3, 4, 2)
            x = x.cpu().numpy()
            if x.dtype != np.uint8:
                raise TypeError(f"{g.path!r}: images must be uint8, got {x.dtype}")
            return x

        def vectors(g: Group) -> Any:
            x = batch[g.path].cpu().numpy().astype(np.float32, copy=False)
            return g.take(x[..., None] if x.ndim == 2 else x)  # a (B, T) scalar leaf is one column

        video = {g.name: images(g) for g in self.groups["video"]}
        state = {g.name: vectors(g) for g in self.groups["state"]}
        action = {g.name: vectors(g) for g in self.groups["action"]}
        if self.language_path is not None and self.language_path in set(batch.keys()):
            texts = [str(t) for t in batch[self.language_path]]
        else:
            texts = [self.language_default] * n
        return [
            VLAStepData(
                images={k: list(v[i]) for k, v in video.items()},
                states={k: v[i] for k, v in state.items()},
                actions={k: v[i] for k, v in action.items()},
                text=texts[i],
                embodiment=embodiment_tag,
            )
            for i in range(n)
        ]

    def to_model_batch(self, batch: Any, processor: Any, embodiment_tag: Any) -> dict:
        """``batch`` -> steps -> ``processor`` -> collated model inputs (what ``model(**inputs)`` takes)."""
        from gr00t.data.types import MessageType

        features = [
            processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            for step in self.to_steps(batch, embodiment_tag)
        ]
        return processor.collator(features)

    # ------------------------------------------------------------------ statistics

    def statistics(self, ds: Any, embodiment_tag: str, exact: bool = True) -> dict:
        """What ``Gr00tN1d7Processor.set_statistics`` takes, computed from ``ds`` in memory.

        Per-leaf statistics come from ``ds.generate_stats`` over exactly the episodes ``ds`` holds
        (pass the training loader, so held-out episodes do not leak); the selection splits them into
        groups in order, and ``RELATIVE`` groups get ``ds.generate_relative_action_stats``.
        Nothing is read from or written to a dataset's own metadata files.

        Raises:
            ValueError: a selected leaf is not a float signal of ``ds`` (valid leaves listed), or a
                relative group's reference state does not fit.
            NotImplementedError: a ``DELTA`` group, or a ``RELATIVE`` group whose reference state is
                a column slice (robotdataset takes a whole state leaf there).
        """
        leaves = sorted({g.path for r in ("state", "action") for g in self.groups[r]})
        info = self.info.with_stats(ds, exact=exact, keys=leaves)
        stats = {p: info[p].stats for p in leaves}
        dims = {p: len(stats[p]["mean"]) for p in leaves}

        state_meta = {
            g.name: {"original_key": g.path, **dict(zip(("start", "end"), g.columns(dims[g.path])))}
            for g in self.groups["state"]
        }
        relative = self._relative_action_stats(ds, dims, exact)
        merged: Dict[str, Any] = {"state": None, "action": {}}
        for path in sorted({g.path for g in self.groups["action"]}):  # one export per action leaf
            sel = {
                g.name: dict(zip(("start", "end"), g.columns(dims[path])))
                for g in self.groups["action"]
                if g.path == path
            }
            part = to_gr00t_statistics(
                stats,
                embodiment_tag,
                modality_meta={"state": state_meta, "action": sel},
                action_key=path,
                relative_action=relative or None,
            )[embodiment_tag]
            merged["state"] = part["state"]
            merged["action"].update(part["action"])
            if "relative_action" in part:
                merged["relative_action"] = part["relative_action"]
        merged["action"] = {g.name: merged["action"][g.name] for g in self.groups["action"]}
        result = {embodiment_tag: merged}
        check_against(result, {r: [g.name for g in self.groups[r]] for r in ("state", "action")})
        self._warn_degenerate(merged)
        return result

    def configure_processor(self, processor: Any, ds: Any, embodiment_tag: Any, exact: bool = True) -> dict:
        """``processor.set_statistics(self.statistics(ds, ...), override=True)``; returns the statistics."""
        tag = getattr(embodiment_tag, "value", embodiment_tag)
        stats = self.statistics(ds, tag, exact=exact)
        processor.set_statistics(stats, override=True)
        return stats

    @staticmethod
    def _warn_degenerate(merged: Mapping[str, Any]) -> None:
        import warnings

        for role in ("state", "action"):
            for g, s in merged[role].items():
                flat = [i for i, (lo, hi) in enumerate(zip(s["q01"], s["q99"])) if hi - lo < 1e-8]
                if flat:
                    warnings.warn(
                        f"{role} group {g!r}: columns {flat} have a zero q01..q99 range; percentile "
                        "normalisation cannot scale them",
                        stacklevel=3,
                    )

    def _relative_action_stats(self, ds: Any, dims: Mapping[str, int], exact: bool) -> dict:
        out: Dict[str, Any] = {}
        state_groups = {g.name: g for g in self.groups["state"]}
        for path in sorted({g.path for g in self.groups["action"]}):
            groups, state_keys, representation = {}, {}, {}
            for g in self.groups["action"]:
                if g.path != path or g.rep == "ABSOLUTE":
                    continue
                if g.rep != "RELATIVE":
                    raise NotImplementedError(f"action group {g.name!r}: rep {g.rep} has no statistics in robotdataset (only ABSOLUTE and RELATIVE)")
                kind = _RELATIVE_REPRESENTATION.get((g.type, g.format))
                if kind is None:
                    raise NotImplementedError(
                        f"action group {g.name!r}: RELATIVE statistics support type/format {sorted(_RELATIVE_REPRESENTATION)}, got ({g.type}, {g.format})"
                    )
                ref = state_groups[g.state_key]
                if ref.is_slice:
                    raise NotImplementedError(
                        f"action group {g.name!r}: its reference state {g.state_key!r} is the column slice "
                        f"{ref.path}[{ref.start}:{ref.end}]; robotdataset's relative-action statistics take the "
                        "reference state as a whole TED leaf of the same width as the group"
                    )
                cols = g.columns(dims[path])
                if dims[ref.path] != cols[1] - cols[0]:
                    raise ValueError(
                        f"action group {g.name!r} has {cols[1] - cols[0]} columns but its reference state "
                        f"{g.state_key!r} ({ref.path}) has {dims[ref.path]}"
                    )
                groups[g.name], state_keys[g.name], representation[g.name] = cols, ref.path, kind
            if groups:
                out.update(
                    ds.generate_relative_action_stats(
                        groups=groups,
                        state_keys=state_keys,
                        horizon=self.action_horizon,
                        representation=representation,
                        action_key=path,
                        exact=exact,
                    )
                )
        return out
