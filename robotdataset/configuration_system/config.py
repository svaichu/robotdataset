"""Fluent, self-learning config builder for WorldModel/VLA training.

`Config` groups fields under top-level categories (e.g. `dataset`,
`training`) and generates group methods dynamically. A group/field must
exist — either loaded from a file or explicitly registered with
`define()` — before it can be set through the fluent API; setting an
unknown group or field raises rather than silently creating it::

    cfg = Config()
    cfg.define("dataset", "name", default="oxe")
    cfg.define("dataset", "batch_size", default=32)
    cfg.dataset(name="oxe", batch_size=32)

Loading a YAML/JSON file teaches the config its groups, field names, and
each field's `type`/`default`. The file doesn't need to carry any
hyperparameter opt info (`bounds`/`values`) — that's typically attached
afterward via `set_bounds`/`set_values`, which is what makes a field
eligible for W&B sweep export::

    cfg = Config.from_file("config.yaml")
    cfg.set_bounds("training", "learning_rate", min=1e-5, max=1e-2)
    cfg.set_values("training", "optimizer", ["adam", "sgd"])
    cfg.to_sweep_file("sweep.yaml", method="bayes", metric={"name": "loss", "goal": "minimize"})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from .field import FieldSpec, infer_type
from .group import Group

PathLike = Union[str, Path]


def _require_yaml():
    """Import pyyaml lazily so plain dict/JSON config use needs no extra deps."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML config support requires pyyaml. Install it with "
            "'pip install robotdataset[config]' or 'pip install pyyaml'."
        ) from exc
    return yaml


class Config:
    """A dynamically-learned, fluent config object."""

    def __init__(self) -> None:
        object.__setattr__(self, "_groups", {})  # group -> {field: value}
        object.__setattr__(self, "_schema", {})  # group -> {field: FieldSpec}

    # -- fluent group access -------------------------------------------------

    def __getattr__(self, name: str) -> Group:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._groups:
            raise AttributeError(
                f"Unknown group '{name}'; define it with Config.define('{name}', ...) "
                f"or load it from a file first"
            )
        return Group(name, self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._update_group_strict(name, value if isinstance(value, dict) else {"value": value})

    def groups(self) -> list[str]:
        return list(self._groups.keys())

    def fields(self, group: str) -> list[str]:
        return list(self._groups.get(group, {}).keys())

    def schema(self, group: str, field: str) -> Optional[FieldSpec]:
        return self._schema.get(group, {}).get(field)

    # -- explicit registration ------------------------------------------------

    def define(self, group: str, field: str, default: Any = None, type: Optional[str] = None, **extra: Any) -> "Config":
        """Register a new field (creating its group if needed) so it can be set via the fluent API."""
        ftype = type if type is not None else (infer_type(default) if default is not None else "str")
        spec_dict: dict = {"type": ftype, "default": default, **extra}
        self._update_group(group, {field: spec_dict})
        return self

    # -- internal learning (used when loading files / defining fields) --------

    def _update_group(self, group: str, fields: dict) -> None:
        self._groups.setdefault(group, {})
        self._schema.setdefault(group, {})
        for name, value in fields.items():
            self._set_field(group, name, value)

    def _set_field(self, group: str, field: str, value: Any) -> None:
        if isinstance(value, dict) and "type" in value:
            spec = FieldSpec.from_spec_dict(field, value)
            self._schema[group][field] = spec
            self._groups[group][field] = spec.default
        else:
            self._schema[group][field] = FieldSpec(name=field, type=infer_type(value), default=value)
            self._groups[group][field] = value

    # -- internal strict updates (used by the fluent group API) ---------------

    def _update_group_strict(self, group: str, fields: dict) -> None:
        if group not in self._groups:
            raise KeyError(
                f"Unknown group '{group}'; define it with Config.define('{group}', ...) "
                f"or load it from a file first"
            )
        for name, value in fields.items():
            self._set_field_strict(group, name, value)

    def _set_field_strict(self, group: str, field: str, value: Any) -> None:
        if field not in self._schema.get(group, {}):
            raise KeyError(
                f"Unknown field '{group}.{field}'; define it with "
                f"Config.define('{group}', '{field}', ...) or load it from a file first"
            )
        if isinstance(value, dict) and "type" in value:
            spec = FieldSpec.from_spec_dict(field, value)
            self._schema[group][field] = spec
            self._groups[group][field] = spec.default
        else:
            self._groups[group][field] = value

    def _require_field(self, group: str, field: str) -> FieldSpec:
        try:
            return self._schema[group][field]
        except KeyError:
            raise KeyError(
                f"Unknown field '{group}.{field}'; define it with "
                f"Config.define('{group}', '{field}', ...) or load it from a file first"
            ) from None

    # -- hyperparameter opt settings --------------------------------------

    def set_bounds(self, group: str, field: str, min: Any = None, max: Any = None, **extra: Any) -> "Config":
        """Attach a continuous search range to an already-known field."""
        spec = self._require_field(group, field)
        bounds: dict = {}
        if min is not None:
            bounds["min"] = min
        if max is not None:
            bounds["max"] = max
        bounds.update(extra)
        spec.bounds = bounds
        return self

    def set_values(self, group: str, field: str, values: list) -> "Config":
        """Attach a discrete/categorical set of candidate values to a field."""
        spec = self._require_field(group, field)
        spec.values = list(values)
        return self

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        cfg = cls()
        for group, fields in data.items():
            if not isinstance(fields, dict):
                continue
            cfg._update_group(group, fields)
        return cfg

    @classmethod
    def from_yaml(cls, path: PathLike) -> "Config":
        yaml = _require_yaml()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: PathLike) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: PathLike) -> "Config":
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        raise ValueError(f"Unsupported config file extension: {path.suffix}")

    # -- exporting ----------------------------------------------------------

    def to_dict(self) -> dict:
        out: dict = {}
        for group, fields in self._groups.items():
            out[group] = {}
            for field, value in fields.items():
                spec = self._schema[group].get(field)
                if spec is not None and spec.is_sweepable():
                    out[group][field] = spec.to_dict()
                else:
                    out[group][field] = value
        return out

    def save(self, path: PathLike) -> None:
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            yaml = _require_yaml()
            with open(path, "w") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported config file extension: {path.suffix}")

    # -- W&B sweep export -----------------------------------------------------

    def to_sweep(
        self,
        method: str = "bayes",
        metric: Optional[dict] = None,
        groups: Optional[list[str]] = None,
    ) -> dict:
        """Build a W&B-compatible sweep config from sweepable fields.

        Fields with `bounds` become continuous ranges (`min`/`max`), fields
        with `values` become discrete/categorical choices, and any other
        field is exported as a fixed `value` from the current config.
        """
        parameters: dict = {}
        for group, fields in self._schema.items():
            if groups is not None and group not in groups:
                continue
            for field, spec in fields.items():
                key = f"{group}.{field}"
                if spec.bounds is not None:
                    parameters[key] = dict(spec.bounds)
                elif spec.values is not None:
                    parameters[key] = {"values": list(spec.values)}
                else:
                    value = self._groups.get(group, {}).get(field)
                    parameters[key] = {"value": value}

        sweep_config: dict = {"method": method, "parameters": parameters}
        if metric is not None:
            sweep_config["metric"] = metric
        return sweep_config

    def to_sweep_file(
        self,
        path: PathLike,
        method: str = "bayes",
        metric: Optional[dict] = None,
        groups: Optional[list[str]] = None,
    ) -> None:
        yaml = _require_yaml()
        sweep_config = self.to_sweep(method=method, metric=metric, groups=groups)
        with open(path, "w") as f:
            yaml.safe_dump(sweep_config, f, sort_keys=False)

    # -- misc -----------------------------------------------------------

    def __repr__(self) -> str:
        if not self._groups:
            return "Config()"

        lines = ["Config"]
        for group, fields in self._groups.items():
            lines.append(f"  {group}:")
            if not fields:
                lines.append("    (empty)")
                continue
            for name, value in fields.items():
                spec = self._schema[group].get(name)
                if spec is not None and spec.is_sweepable():
                    extra = ", ".join(
                        f"{k}={v}"
                        for k, v in (("bounds", spec.bounds), ("values", spec.values))
                        if v is not None
                    )
                    lines.append(f"    {name}: {value!r} [{extra}]")
                else:
                    ftype = spec.type if spec is not None else infer_type(value)
                    lines.append(f"    {name}: {value!r} ({ftype})")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.to_dict() == other.to_dict()
