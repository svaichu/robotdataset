"""argparse integration for `Config`.

Every known field is exposed as a dotted `--<group>.<field>` option — the
same keys the W&B sweep export uses, so a `wandb agent` command line
(`python train.py --training.learning_rate=0.001`) parses directly.
Values are converted using each field's learned `type`, and options the
user didn't pass are suppressed so the config keeps its file/default
values (precedence: defaults < config file < CLI).
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from .config import Config
    from .field import FieldSpec

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def str2bool(value: Union[str, bool]) -> bool:
    """Parse the boolean spellings used on training-script command lines."""
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (true/false), got {value!r}")


def parse_list(value: str) -> list:
    """Parse a list from a JSON array ('[1, 2]') or comma-separated string ('a,b,c')."""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass
    items = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(json.loads(item))
        except ValueError:
            items.append(item)
    return items


def parse_dict(value: str) -> dict:
    """Parse a dict from a JSON object string ('{"a": 1}')."""
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}")
    return parsed


_CONVERTERS: dict[str, Callable[[str], Any]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": str2bool,
    "list": parse_list,
    "dict": parse_dict,
}


def converter_for(spec: "FieldSpec") -> Callable[[str], Any]:
    """Return the string→value converter for a field's learned type."""
    return _CONVERTERS.get(spec.type, str)


_TYPE_NAME_ALIASES: dict[type, str] = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
    list: "list",
    dict: "dict",
}


def normalize_type(type_: Union[str, type, None]) -> Optional[str]:
    """Map a Python type (as passed to `add_argument`) to its schema type name.

    Accepts the schema's own string names (`"float"`) unchanged, so callers
    can mix `add_argument(..., type=float)` with `define(..., type="float")`.
    """
    if type_ is None or isinstance(type_, str):
        return type_
    try:
        return _TYPE_NAME_ALIASES[type_]
    except (KeyError, TypeError):
        raise TypeError(
            f"Unsupported type {type_!r} for add_argument; pass one of "
            "int, float, str, bool, list, dict (or their string names)"
        ) from None


def add_config_arguments(
    cfg: "Config",
    parser: argparse.ArgumentParser,
    groups: Optional[list[str]] = None,
) -> argparse.ArgumentParser:
    """Add one `--<group>.<field>` option per known field to `parser`.

    Options default to `argparse.SUPPRESS`, so only flags the user actually
    passed show up in the parsed namespace — everything else keeps its
    current config value. Fields with `values` become `choices`; `bool`
    fields accept both a bare flag (`--training.shuffle`) and an explicit
    value (`--training.shuffle=false`).
    """
    for group in cfg.groups():
        if groups is not None and group not in groups:
            continue
        arg_group = parser.add_argument_group(group)
        for field in cfg.fields(group):
            spec = cfg.schema(group, field)
            key = f"{group}.{field}"
            current = cfg._groups[group][field]
            kwargs: dict = {
                "type": converter_for(spec),
                "default": argparse.SUPPRESS,
                "dest": key,
                "help": spec.help if spec.help else f"({spec.type}) default: {current!r}",
            }
            if spec.type == "bool":
                kwargs["nargs"] = "?"
                kwargs["const"] = True
            if spec.values is not None and spec.type in ("int", "float", "str"):
                kwargs["choices"] = spec.values
            else:
                kwargs["metavar"] = spec.type.upper()
            arg_group.add_argument(f"--{key}", **kwargs)
    return parser


def apply_namespace(cfg: "Config", args: Union[argparse.Namespace, dict]) -> "Config":
    """Apply dotted `group.field` entries from a namespace/dict onto `cfg`.

    Keys without a dot (e.g. `config`, `method`) are ignored so a shared
    parser can carry script-level options alongside config overrides.
    String values for non-str fields are coerced through the field's
    converter, so plain string dicts (e.g. `wandb.config`) work too.
    Unknown dotted keys raise, matching the fluent API's strictness.
    """
    values = args if isinstance(args, dict) else vars(args)
    for key, value in values.items():
        if "." not in key:
            continue
        group, _, field = key.partition(".")
        spec = cfg._require_field(group, field)
        if isinstance(value, str) and spec.type != "str":
            value = converter_for(spec)(value)
        cfg._update_group_strict(group, {field: value})
    return cfg
