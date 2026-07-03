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

    `bounds`/`values` are only populated for fields declared as
    hyperparameters (e.g. `{type: float, bounds: {min: 1e-5, max: 1e-2}}`),
    which is what makes a field eligible for W&B sweep export.
    """

    name: str
    type: str
    bounds: Optional[dict] = None
    values: Optional[list] = None
    default: Any = None

    def is_sweepable(self) -> bool:
        return self.bounds is not None or self.values is not None

    @classmethod
    def from_spec_dict(cls, name: str, spec: dict) -> "FieldSpec":
        return cls(
            name=name,
            type=spec.get("type", "str"),
            bounds=spec.get("bounds"),
            values=spec.get("values"),
            default=spec.get("default"),
        )

    def to_dict(self) -> dict:
        out: dict = {"type": self.type}
        if self.bounds is not None:
            out["bounds"] = self.bounds
        if self.values is not None:
            out["values"] = self.values
        if self.default is not None:
            out["default"] = self.default
        return out
