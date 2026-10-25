"""Dataset info: a model-agnostic description of what a robotdataset loader holds.

Every loader samples TED batches, a flat ``TensorDict`` keyed by ``"/"``-separated paths.
:class:`DatasetInfo` lists those paths as :class:`ModalityInfo` entries (kind, dtype, shape),
hides the TED bookkeeping (``next/...``, ``collector/...``, ``done``, ...) and papers over the
differences between loaders (``ds.modalities`` is grouped by kind on OXE but path-to-spec
elsewhere; ``ds.image_keys`` holds tuples on some loaders and strings on others).

The info is also the editable hand-off to a model adapter: :meth:`DatasetInfo.to_yaml`
writes the loader, its arguments and one entry per modality; you edit the entries
(``use``, order, ``name``, ``role``, ``split``) and an adapter in
:mod:`robotdataset.adapters` reads the file back.  Statistics are *not* part of the file:
they depend on which episodes a loader holds, so they are computed on demand
(:meth:`DatasetInfo.with_stats`) and live in memory.

Command line::

    python -m robotdataset.info --loader OXEDataset --kwargs '{"dataset_name": "viola"}' \\
        --episodes 0:2 --stats -o viola.yaml

Python::

    info = DatasetInfo.from_loader("LeRobotVideoDataset", {"local_path": "..."}, episodes=[0])
    print(info.table())
    info.to_yaml("modalities.yaml")           # edit: set use: true on the modalities you want
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOADERS = ("LeRobotVideoDataset", "Table30v2Dataset", "OXEDataset", "AgiBotWorldBetaDataset", "StationParquetDataset")
KINDS = ("image", "state", "action", "text", "generic")
ROLES = ("video", "state", "action", "language")
_ROLE_OF_KIND = {"image": "video", "state": "state", "action": "action", "text": "language"}

_BOOKKEEPING_PREFIXES = ("next/", "collector/")
_BOOKKEEPING_KEYS = frozenset(
    {"done", "terminated", "truncated", "reward", "discount", "is_first", "is_last", "is_terminal", "episode_id"}
)
ADAPTERS = {"gr00t": "robotdataset.adapters.gr00t:Gr00tAdapter"}  # name -> "module:Class", imported lazily
_RESERVED_KWARGS = ("episodes", "batch_size")
_ENTRY_FIELDS = {"path", "kind", "dtype", "shape", "use", "name", "role", "split"}
_SPLIT_FIELDS = {"name", "columns"}


# --------------------------------------------------------------------------- helpers


def _path(key: Any) -> str:
    return "/".join(key) if isinstance(key, tuple) else str(key)


def _is_bookkeeping(path: str) -> bool:
    return path.startswith(_BOOKKEEPING_PREFIXES) or path in _BOOKKEEPING_KEYS


def _dtype_name(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "").replace("tf.", "")


def _shape(raw: Any) -> Tuple[int, ...]:
    """Tuple of ints from a tuple/list, a ``"(7,)"`` string or ``None`` (scalar)."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return tuple(int(d) for d in raw)


def _holds_strings(leaf: Any) -> bool:
    """A non-tensor batch leaf whose items are Python strings (for example a language instruction per sample)."""
    import torch

    if isinstance(leaf, torch.Tensor):
        return False
    try:
        items = list(leaf)
    except TypeError:
        return False
    return bool(items) and all(isinstance(getattr(x, "data", x), str) for x in items)


def _is_float(dtype: Optional[str]) -> bool:
    return dtype is not None and dtype.startswith(("float", "bfloat"))


def _default_names(paths: Sequence[str]) -> Dict[str, str]:
    """Shortest unique name per path: the last component, extended leftwards on collision."""
    parts = {p: p.split("/") for p in paths}
    depth = {p: 1 for p in paths}

    def name(p: str) -> str:
        return "_".join(parts[p][-depth[p]:])

    while True:
        by_name: Dict[str, List[str]] = {}
        for p in paths:
            by_name.setdefault(name(p), []).append(p)
        grown = False
        for ps in by_name.values():
            if len(ps) > 1:
                for p in ps:
                    if depth[p] < len(parts[p]):
                        depth[p] += 1
                        grown = True
        if not grown:
            return {p: name(p) for p in paths}


# --------------------------------------------------------------------------- YAML writing


def _flow(obj: Any) -> str:
    """``obj`` on one line of flow-style YAML (scalars are quoted where flow context needs it)."""
    import yaml

    return yaml.safe_dump(obj, default_flow_style=True, width=10**6, sort_keys=False).strip()


def _cell(key: str, value: Any) -> str:
    """``key: value`` as it appears inside a flow mapping."""
    return _flow({key: value})[1:-1]


def _line(key: str, value: Any) -> str:
    """``key: value`` on one line: a list or mapping in flow style, a scalar as is."""
    return f"{key}: {_flow(value)}" if isinstance(value, (list, tuple, Mapping)) else _cell(key, value)


def _flow_rows(rows: Sequence[Mapping[str, Any]], first: Sequence[str] = ()) -> List[str]:
    """Every mapping as one ``{key: value, ...}`` line; the keys all rows share line up in columns."""
    if not rows:
        return []
    order = list(first) + [k for k in rows[0] if k not in first]
    common = [k for k in order if all(k in r for r in rows)]
    width = {k: max(len(_cell(k, r[k])) for r in rows) for k in common}
    lines = []
    for r in rows:
        extras = [_cell(k, v) for k, v in r.items() if k not in width]
        parts = []
        for i, k in enumerate(common):
            cell = _cell(k, r[k])
            # the last shared column is not padded, so extras (role, split, ...) follow it directly
            parts.append((cell + ",").ljust(width[k] + 1) if i < len(common) - 1 else cell + ("," if extras else ""))
        lines.append("{" + " ".join(parts + ([", ".join(extras)] if extras else [])) + "}")
    return lines


def _modality_lines(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    """The ``modalities:`` list: ``use`` first and columns aligned, a blank line between kinds."""
    rows = _flow_rows(entries, first=("use", "path", "kind", "dtype", "shape", "name"))
    out: List[str] = []
    for i, (row, e) in enumerate(zip(rows, entries)):
        if i and e["kind"] != entries[i - 1]["kind"]:
            out.append("")
        out.append(f"  - {row}")
    return out


def _align_notes(rows: Sequence[Tuple[str, Optional[str]]]) -> List[str]:
    """Write each ``(code, note)`` as ``code  # note``; consecutive annotated lines of one indentation share a note column."""
    out: List[str] = []
    i = 0
    while i < len(rows):
        if rows[i][1] is None:
            out.append(rows[i][0])
            i += 1
            continue
        indent = len(rows[i][0]) - len(rows[i][0].lstrip())
        j = i
        while j < len(rows) and rows[j][1] is not None and len(rows[j][0]) - len(rows[j][0].lstrip()) == indent:
            j += 1
        width = max(len(code) for code, _ in rows[i:j])
        out.extend(f"{code.ljust(width)}  # {note}" for code, note in rows[i:j])
        i = j
    return out


def _block_rows(
    block: Mapping[str, Any],
    indent: int,
    notes: Mapping[str, str],
    examples: Mapping[str, Sequence[str]],
    prefix: str,
) -> List[Tuple[str, Optional[str]]]:
    pad = " " * indent
    rows: List[Tuple[str, Optional[str]]] = []
    for key, value in block.items():
        path = f"{prefix}{key}"
        note = notes.get(path)
        if isinstance(value, Mapping) and not value:
            rows.append((f"{pad}{key}: {{}}", note))
            rows.extend((f"{pad}# {ln}", None) for ln in examples.get(path, ()))
        elif isinstance(value, Mapping) and all(isinstance(v, Mapping) for v in value.values()):
            rows.append((f"{pad}{key}:", note))  # a table of groups: one aligned flow row per entry
            names = [f"{k}:" for k in value]
            width = max(len(n) for n in names)
            rows.extend((f"{pad}  {n.ljust(width)} {row}", None) for n, row in zip(names, _flow_rows(list(value.values()))))
        elif isinstance(value, Mapping):
            rows.append((f"{pad}{key}:", note))
            rows.extend(_block_rows(value, indent + 2, notes, examples, f"{path}."))
        else:
            rows.append((f"{pad}{_line(key, value)}", note))
    return rows


def pretty_block(
    block: Mapping[str, Any],
    indent: int = 2,
    notes: Optional[Mapping[str, str]] = None,
    examples: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[str]:
    """A model block as readable YAML lines.

    Nested mappings are blocks, lists and leaf values stay on one line (``video_delta: [0]``), and a mapping whose values
    are all mappings (a table of groups) becomes one aligned ``{...}`` row per entry.  ``notes`` maps a dotted key
    (``language.default``) to a comment written after it; ``examples`` maps the dotted key of an empty mapping to
    comment lines that show what an entry looks like.
    """
    return _align_notes(_block_rows(block, indent, notes or {}, examples or {}, ""))


def adapter_class(name: str) -> Optional[type]:
    """The model adapter class registered under ``name`` in :data:`ADAPTERS` (``None`` if unknown or unimportable)."""
    import importlib

    spec = ADAPTERS.get(name)
    if spec is None:
        return None
    module, _, cls_name = spec.partition(":")
    try:
        return getattr(importlib.import_module(module), cls_name)
    except ImportError:
        return None


# --------------------------------------------------------------------------- entries


@dataclass(frozen=True)
class ModalityInfo:
    """One TED leaf of a loader.

    ``path``, ``kind``, ``dtype`` and ``shape`` (per step; ``()`` is a scalar, images are HWC as
    stored) are facts.  ``use``, ``name``, ``role`` and ``split`` are the editable selection a
    model adapter reads; ``stats`` is in-memory only.
    """

    path: str
    kind: str = "generic"
    dtype: Optional[str] = None
    shape: Tuple[int, ...] = ()
    use: bool = False
    name: Optional[str] = None
    role: Optional[str] = None
    split: Optional[Tuple[Tuple[str, Tuple[int, int]], ...]] = None
    stats: Optional[Mapping[str, Any]] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.path}: kind {self.kind!r} is not one of {KINDS}")
        if self.role is not None and self.role not in ROLES:
            raise ValueError(f"{self.path}: role {self.role!r} is not one of {ROLES}")

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    @property
    def is_string(self) -> bool:
        """A text leaf (a Python string per step), as opposed to a float embedding of text."""
        return self.kind == "text" and not _is_float(self.dtype)

    @property
    def looks_like_image(self) -> bool:
        """``kind == "image"``, or a uint8 HWC leaf whose name robotdataset's heuristic did not flag."""
        return self.is_image or (self.dtype == "uint8" and len(self.shape) == 3 and self.shape[-1] in (1, 3, 4))

    @property
    def is_signal(self) -> bool:
        """A float, non-image leaf: statistics and state/action roles apply."""
        return _is_float(self.dtype) and not self.is_image

    @property
    def width(self) -> Optional[int]:
        """Columns of a vector leaf (``1`` for a scalar); ``None`` if it is not a plain vector."""
        if any(d < 0 for d in self.shape) or len(self.shape) > 1:
            return None
        return self.shape[0] if self.shape else 1

    @property
    def effective_role(self) -> Optional[str]:
        return self.role or _ROLE_OF_KIND.get(self.kind)


# --------------------------------------------------------------------------- info


@dataclass(frozen=True)
class DatasetInfo:
    """What a loader holds, plus the editable selection (see the module docstring)."""

    loader: str
    modalities: Tuple[ModalityInfo, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    episodes_loaded: Optional[int] = None
    steps_loaded: Optional[int] = None
    fps: Optional[float] = None
    extra: Mapping[str, Any] = field(default_factory=dict)  # adapter blocks, e.g. ``gr00t:``

    # ------------------------------------------------------------ construction

    @classmethod
    def from_dataset(
        cls,
        ds: Any,
        loader: Optional[str] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        stats: bool = False,
        exact: bool = False,
    ) -> "DatasetInfo":
        """Describe a built loader.

        Args:
            ds: Any torch-path robotdataset loader.
            loader: Loader class name to record (default: ``type(ds).__name__``).
            kwargs: The loader's constructor arguments to record, minus ``episodes`` and
                ``batch_size`` (loaders do not remember them).
            stats: Also compute per-leaf statistics over the episodes ``ds`` holds.
            exact: Exact quantiles for those statistics.
        """
        batch = ds.sample()
        present = set(batch.keys())  # the flat paths the sampler really yields
        entries: List[ModalityInfo] = []
        for key, spec in ds.get_modalities().items():
            path = _path(key)
            if _is_bookkeeping(path) or path not in present or not isinstance(spec, Mapping):
                continue
            kind = spec.get("kind") or "generic"
            shape = _shape(spec.get("shape"))
            leaf = batch[path]
            if any(d < 0 for d in shape) and hasattr(leaf, "shape") and len(leaf.shape) >= 2:
                shape = tuple(int(d) for d in leaf.shape[2:])  # metadata has variable dims: use the sampled (B, T, ...) leaf
            entries.append(
                ModalityInfo(
                    path=path,
                    kind=kind if kind in KINDS else "generic",
                    dtype=_dtype_name(spec.get("dtype")),
                    shape=shape,
                )
            )
        listed = {e.path for e in entries}
        for path in sorted(present - listed):  # string leaves the loader yields but get_modalities() does not describe
            if not _is_bookkeeping(path) and _holds_strings(batch[path]):
                entries.append(ModalityInfo(path=path, kind="text", dtype="string", shape=()))
        entries.sort(key=lambda e: KINDS.index(e.kind))  # images, states, actions, text; loader order within a kind
        names = _default_names([e.path for e in entries])
        fps = getattr(ds, "fps", None)
        info = cls(
            loader=loader or type(ds).__name__,
            modalities=tuple(replace(e, name=names[e.path]) for e in entries),
            kwargs=dict(kwargs or {}),
            episodes_loaded=int(ds.num_episodes),
            steps_loaded=int(len(ds)),
            fps=None if fps is None else float(fps),
        )
        return info.with_stats(ds, exact=exact) if stats else info

    @classmethod
    def from_loader(
        cls,
        loader: str,
        kwargs: Optional[Mapping[str, Any]] = None,
        episodes: Optional[Sequence[int]] = None,
        stats: bool = False,
        exact: bool = False,
    ) -> "DatasetInfo":
        """Build the named loader (a few episodes are enough to list the modalities) and describe it."""
        ds = build_loader(loader, kwargs or {}, episodes=episodes, batch_size=2)
        return cls.from_dataset(ds, loader=loader, kwargs=kwargs, stats=stats, exact=exact)

    # ------------------------------------------------------------ queries

    def __getitem__(self, path: str) -> ModalityInfo:
        for m in self.modalities:
            if m.path == path:
                return m
        raise KeyError(f"{path!r} is not a modality of this dataset; valid paths: {self.paths}")

    @property
    def paths(self) -> List[str]:
        return [m.path for m in self.modalities]

    @property
    def signal_paths(self) -> List[str]:
        return [m.path for m in self.modalities if m.is_signal]

    @property
    def selected(self) -> List[ModalityInfo]:
        """The modalities with ``use: true``, in file order."""
        return [m for m in self.modalities if m.use]

    # ------------------------------------------------------------ statistics (in memory)

    def with_stats(self, ds: Any, exact: bool = False, keys: Optional[Iterable[str]] = None) -> "DatasetInfo":
        """Attach ``ds.generate_stats`` (over the episodes ``ds`` holds) to float, non-image leaves.

        ``keys`` restricts the leaves (default: every float, non-image leaf); the others keep
        ``stats=None``.  Exact quantiles rescan the data, so restrict ``keys`` when only a few matter.
        """
        keys = self.signal_paths if keys is None else list(keys)
        stats = ds.generate_stats(keys=keys, exact=exact) if keys else {}
        return replace(self, modalities=tuple(replace(m, stats=stats.get(m.path)) for m in self.modalities))

    # ------------------------------------------------------------ display

    def table(self) -> str:
        """Fixed-width overview; statistics columns appear when they were computed."""
        has_stats = any(m.stats for m in self.modalities)
        head = ["path", "kind", "dtype", "shape", "use", "name"] + (["std min..max", "const dims"] if has_stats else [])
        rows = []
        for m in self.modalities:
            row = [m.path, m.kind, m.dtype or "-", str(list(m.shape)), "yes" if m.use else "no", m.name or ""]
            if has_stats:
                if m.stats:
                    std = [float(s) for s in m.stats["std"]]
                    const = sum(1 for s in std if s < 1e-8)
                    row += [f"{min(std):.3g}..{max(std):.3g}", f"{const}/{len(std)}" if const else "-"]
                else:
                    row += ["", ""]
            rows.append(row)
        widths = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]

        def line(r: Sequence[Any]) -> str:
            return "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()

        meta = f"{self.loader}: {self.episodes_loaded} episode(s), {self.steps_loaded} step(s)" + (
            f", {self.fps:g} fps" if self.fps else ""
        )
        return "\n".join([meta, line(head), line(["-" * w for w in widths])] + [line(r) for r in rows])

    # ------------------------------------------------------------ YAML round trip

    def to_dict(self) -> Dict[str, Any]:
        entries = []
        for m in self.modalities:
            e: Dict[str, Any] = {"path": m.path, "kind": m.kind}
            if m.dtype is not None:
                e["dtype"] = m.dtype
            e["shape"] = list(m.shape)
            e["use"] = m.use
            e["name"] = m.name
            if m.role is not None:
                e["role"] = m.role
            if m.split:
                e["split"] = [{"name": n, "columns": list(c)} for n, c in m.split]
            entries.append(e)
        dataset: Dict[str, Any] = {"loader": self.loader, "kwargs": dict(self.kwargs)}
        for k in ("episodes_loaded", "steps_loaded", "fps"):
            if getattr(self, k) is not None:
                dataset[k] = getattr(self, k)
        out: Dict[str, Any] = {"dataset": dataset, "modalities": entries}
        out.update(self.extra)
        return out

    def to_yaml(self, path: Optional[str] = None, comments: Optional[Mapping[str, str]] = None) -> str:
        """The editable file (returned, and written to ``path`` if given).

        Modalities are one aligned line each (``use`` first, a blank line between kinds). A block of a registered model
        adapter (``gr00t:``) is written by that adapter, with its help text and notes; any other block gets a generic
        readable layout.  ``comments`` maps a block name to help text that replaces the adapter's.
        """
        d = self.to_dict()
        lines = [
            "# robotdataset dataset info (generated). Edit it, then load it with a model adapter.",
            "#",
            "#   use    true on every modality the model should get (all start as false)",
            "#   order  the order of this list is the order handed to the model",
            "#   name   the model-side group name (defaults to the shortest unique path suffix)",
            "#   role   optional override of the role implied by kind: video | state | action | language",
            "#   split  optional, cuts one vector leaf into named groups: [{name: eef, columns: [0, 6]}, ...]",
            "#",
            "# kind, dtype and shape describe the data; they are checked against the loader when the file is loaded.",
            "dataset:",
            f"  loader: {self.loader}",
            f"  kwargs: {_flow(dict(self.kwargs))}",
        ]
        for k in ("episodes_loaded", "steps_loaded", "fps"):
            if k in d["dataset"]:
                lines.append(f"  {k}: {d['dataset'][k]}")
        lines.append("")
        lines.append("modalities:")
        lines.extend(_modality_lines(d["modalities"]))
        for key, block in self.extra.items():
            adapter = adapter_class(key)
            if comments is not None and key in comments:
                help_text = comments[key]
            else:
                help_text = adapter.YAML_HELP if adapter is not None else ""
            lines.append("")
            lines.extend(f"# {ln}" if ln else "#" for ln in help_text.splitlines())
            lines.append(f"{key}:")
            lines.extend(adapter.yaml_lines(block) if adapter is not None else pretty_block(block))
        text = "\n".join(lines) + "\n"
        if path is not None:
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetInfo":
        for need in ("dataset", "modalities"):
            if need not in data:
                raise ValueError(f"dataset info: missing top-level '{need}'")
        ds = data["dataset"]
        if "loader" not in ds:
            raise ValueError("dataset info: dataset.loader is missing")
        entries = []
        for i, e in enumerate(data["modalities"]):
            where = f"modalities[{i}]"
            if not isinstance(e, Mapping) or "path" not in e:
                raise ValueError(f"{where}: expected a mapping with a 'path'")
            unknown = sorted(set(e) - _ENTRY_FIELDS)
            if unknown:
                raise ValueError(f"{where} ({e['path']}): unknown field(s) {unknown}; allowed: {sorted(_ENTRY_FIELDS)}")
            split = None
            if e.get("split"):
                parts = []
                for s in e["split"]:
                    cols = s.get("columns")
                    ok = (
                        not set(s) - _SPLIT_FIELDS
                        and "name" in s
                        and isinstance(cols, (list, tuple))
                        and len(cols) == 2
                        and 0 <= cols[0] < cols[1]
                    )
                    if not ok:
                        raise ValueError(
                            f"{where} ({e['path']}): split entries need name and columns [start, end] "
                            f"with 0 <= start < end, got {s}"
                        )
                    parts.append((str(s["name"]), (int(cols[0]), int(cols[1]))))
                split = tuple(parts)
            entries.append(
                ModalityInfo(
                    path=e["path"],
                    kind=e.get("kind", "generic"),
                    dtype=e.get("dtype"),
                    shape=_shape(e.get("shape")),
                    use=bool(e.get("use", False)),
                    name=e.get("name"),
                    role=e.get("role"),
                    split=split,
                )
            )
        paths = [m.path for m in entries]
        dup = sorted({p for p in paths if paths.count(p) > 1})
        if dup:
            raise ValueError(f"dataset info: duplicate modality path(s) {dup}")
        return cls(
            loader=ds["loader"],
            modalities=tuple(entries),
            kwargs=dict(ds.get("kwargs") or {}),
            episodes_loaded=ds.get("episodes_loaded"),
            steps_loaded=ds.get("steps_loaded"),
            fps=ds.get("fps"),
            extra={k: v for k, v in data.items() if k not in ("dataset", "modalities")},
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetInfo":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    # ------------------------------------------------------------ validation

    def check_against(self, fresh: "DatasetInfo", only: Optional[Iterable[str]] = None) -> None:
        """Raise ``ValueError`` listing every entry here that the fresh listing does not match.

        Used when an edited file is loaded: paths must exist and dtype/shape must still be what
        the file recorded (a loader or dataset version that changed would otherwise train
        silently on different data).  ``only`` restricts the check to those paths (the selected ones).
        """
        problems = []
        have = {m.path: m for m in fresh.modalities}
        wanted = None if only is None else set(only)
        for m in self.modalities:
            if wanted is not None and m.path not in wanted:
                continue
            f = have.get(m.path)
            if f is None:
                problems.append(f"{m.path}: not in the loader (valid paths: {sorted(have)})")
                continue
            if m.dtype is not None and f.dtype is not None and m.dtype != f.dtype:
                problems.append(f"{m.path}: dtype {m.dtype} in the file, {f.dtype} in the loader")
            if m.shape != f.shape and not any(d < 0 for d in m.shape + f.shape):
                problems.append(f"{m.path}: shape {list(m.shape)} in the file, {list(f.shape)} in the loader")
        if self.loader != fresh.loader:
            problems.append(f"loader {self.loader} in the file, {fresh.loader} used")
        if problems:
            raise ValueError("dataset info file does not match the loader:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- loaders


def build_loader(
    loader: str,
    kwargs: Mapping[str, Any],
    episodes: Optional[Sequence[int]] = None,
    batch_size: int = 2,
    **extra: Any,
) -> Any:
    """Instantiate the robotdataset loader class ``loader`` with ``kwargs`` (plus episodes/batch size).

    ``extra`` (for example ``delta_timestamps``, ``control_frequency``) is passed to the loader too.
    """
    import robotdataset

    if loader not in LOADERS:
        raise ValueError(f"loader {loader!r} is not one of {list(LOADERS)}")
    cls = getattr(robotdataset, loader, None)
    if cls is None:
        raise ImportError(
            f"robotdataset.{loader} is unavailable (optional dependencies missing); "
            "install the matching extra, e.g. robotdataset[hf] or robotdataset[oxe]"
        )
    clash = [k for k in _RESERVED_KWARGS if k in kwargs]
    if clash:
        raise ValueError(f"kwargs must not set {clash}: they are arguments of the call")
    return cls(**dict(kwargs), episodes=None if episodes is None else list(episodes), batch_size=batch_size, **extra)


# --------------------------------------------------------------------------- CLI


def parse_episodes(spec: Any) -> Optional[List[int]]:
    """Episode list from a compact spec: ``None``, an int, ``"0:46"`` (half-open), ``"0:3,7"`` or a list of those.

    This is how run files write splits (``train_episodes: ["0:46"]``) and how ``--episodes`` is read.
    """
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise TypeError("episodes must be ints or 'a:b' ranges, got a bool")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        out: List[int] = []
        for part in spec.split(","):
            a, sep, b = part.strip().partition(":")
            if not a.strip() and not sep:
                continue
            out.extend(range(int(a), int(b)) if sep else [int(a)])
        return out
    out = []
    for item in spec:
        out.extend(parse_episodes(item) or [])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m robotdataset.info", description="Write the editable dataset-info YAML of a loader."
    )
    p.add_argument("--loader", required=True, choices=LOADERS)
    p.add_argument("--kwargs", default="{}", help='loader arguments as JSON, e.g. \'{"dataset_name": "viola"}\'')
    p.add_argument("--episodes", help="episodes to load for the listing: 0:2 or 0,3,5 (default: the loader's default)")
    p.add_argument("--adapter", choices=sorted(ADAPTERS), help="also write that model adapter's default settings block")
    p.add_argument("--stats", action="store_true", help="also compute and print statistics (not written to the YAML)")
    p.add_argument("--exact", action="store_true", help="exact quantiles for --stats")
    p.add_argument("-o", "--output", help="write the YAML here")
    args = p.parse_args(argv)
    info = DatasetInfo.from_loader(
        args.loader, json.loads(args.kwargs), episodes=parse_episodes(args.episodes), stats=args.stats, exact=args.exact
    )
    print(info.table())
    if args.adapter:
        info = adapter_class(args.adapter).with_defaults(info)
    if args.output:
        info.to_yaml(args.output)
        print(f"\nwrote {args.output}: set 'use: true' on the modalities to feed the model")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
