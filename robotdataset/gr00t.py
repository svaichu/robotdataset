"""GR00T export of dataset statistics.

Kept apart from :mod:`robotdataset.stats` so the core stays GR00T-agnostic.  This
module does not import the ``gr00t`` package.

GR00T's processor takes ``{embodiment_tag: {modality: {group: {stat: [floats]}}}}``
with ``modality`` in ``state``, ``action``, ``relative_action`` and ``stat`` in
``mean, std, min, max, q01, q99``.  Group names must match the ``modality_keys``
of the GR00T ``ModalityConfig``; a group's vector length defines the dimension
used to split a predicted action back into groups.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

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
