"""Field specifications learned from config files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def infer_type(value: Any) -> str:
    """Map a Python value to a config field type name."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


@dataclass
class FieldSpec:
    """Learned metadata for a single config field.

    `type` and `default` are the required subfields: every field ends up
    with both, inferred from a plain YAML/JSON value if the file didn't
    spell them out explicitly.

    `bounds`/`values` are the hyperparameter opt settings that make a
    field eligible for W&B sweep export. The source file may declare them
    directly, but they are typically attached afterward via
    `Config.set_bounds`/`Config.set_values`.
    """

    name: str
    type: str
    default: Any
    bounds: Optional[dict] = None
    values: Optional[list] = None
    help: Optional[str] = None

    def is_sweepable(self) -> bool:
        return self.bounds is not None or self.values is not None

    @classmethod
    def from_spec_dict(cls, name: str, spec: dict) -> "FieldSpec":
        bounds = spec.get("bounds")
        values = spec.get("values")
        default = spec.get("default")
        if default is None:
            if bounds is not None and "min" in bounds:
                default = bounds["min"]
            elif values:
                default = values[0]
        return cls(
            name=name,
            type=spec.get("type", infer_type(default) if default is not None else "str"),
            default=default,
            bounds=bounds,
            values=values,
            help=spec.get("help"),
        )

    def to_dict(self) -> dict:
        out: dict = {"type": self.type, "default": self.default}
        if self.bounds is not None:
            out["bounds"] = self.bounds
        if self.values is not None:
            out["values"] = self.values
        return out
