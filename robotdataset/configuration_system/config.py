"""Fluent, self-learning config builder for WorldModel/VLA training.

`Config` groups fields under top-level categories (e.g. `dataset`,
`training`) and generates group methods dynamically:

    cfg = Config()
    cfg.dataset(name="oxe", batch_size=32).training(learning_rate=1e-4)

Loading a YAML/JSON file teaches the config its groups, field names and
types, and any hyperparameter bounds/values needed for W&B sweep export::

    cfg = Config.from_file("config.yaml")
    cfg.to_sweep_file("sweep.yaml", method="bayes", metric={"name": "loss", "goal": "minimize"})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .field import FieldSpec, infer_type
from .group import Group

PathLike = Union[str, Path]


class Config:
    """A dynamically-learned, fluent config object."""

    def __init__(self) -> None:
        object.__setattr__(self, "_groups", {})  # group -> {field: value}
        object.__setattr__(self, "_schema", {})  # group -> {field: FieldSpec}

    # -- fluent group access -------------------------------------------------

    def __getattr__(self, name: str) -> Group:
        if name.startswith("_"):
            raise AttributeError(name)
        return Group(name, self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._update_group(name, value if isinstance(value, dict) else {"value": value})

    def groups(self) -> list[str]:
        return list(self._groups.keys())

    def fields(self, group: str) -> list[str]:
        return list(self._groups.get(group, {}).keys())

    def schema(self, group: str, field: str) -> Optional[FieldSpec]:
        return self._schema.get(group, {}).get(field)

    # -- internal learning ----------------------------------------------------

    def _update_group(self, group: str, fields: dict) -> None:
        self._groups.setdefault(group, {})
        self._schema.setdefault(group, {})
        for name, value in fields.items():
            self._set_field(group, name, value)

    def _set_field(self, group: str, field: str, value: Any) -> None:
        if isinstance(value, dict) and "type" in value and ("bounds" in value or "values" in value):
            spec = FieldSpec.from_spec_dict(field, value)
            self._schema[group][field] = spec
            self._groups[group][field] = spec.default if spec.default is not None else value
        else:
            self._schema[group][field] = FieldSpec(name=field, type=infer_type(value))
            self._groups[group][field] = value

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
        sweep_config = self.to_sweep(method=method, metric=metric, groups=groups)
        with open(path, "w") as f:
            yaml.safe_dump(sweep_config, f, sort_keys=False)

    # -- misc -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.to_dict() == other.to_dict()
