"""Group proxy returned by `Config.<group_name>` attribute access."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class Group:
    """Callable namespace bound to a single config group.

    `config.dataset(batch_size=32)` updates the group and returns the
    parent `Config` so calls can be chained. `config.dataset.batch_size`
    reads the current value.
    """

    def __init__(self, name: str, config: "Config"):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_config", config)

    def __call__(self, **fields: Any) -> "Config":
        self._config._update_group_strict(self._name, fields)
        return self._config

    def __getattr__(self, field: str) -> Any:
        fields = self._config._groups.get(self._name, {})
        if field not in fields:
            raise AttributeError(
                f"Unknown field '{self._name}.{field}'; define it with "
                f"Config.define('{self._name}', '{field}', ...) or load it from a file first"
            )
        return fields[field]

    def __setattr__(self, field: str, value: Any) -> None:
        self._config._update_group_strict(self._name, {field: value})

    def __getitem__(self, field: str) -> Any:
        return self._config._groups[self._name][field]

    def __setitem__(self, field: str, value: Any) -> None:
        self._config._update_group_strict(self._name, {field: value})

    def __contains__(self, field: str) -> bool:
        return field in self._config._groups.get(self._name, {})

    def __repr__(self) -> str:
        fields = self._config._groups.get(self._name, {})
        return f"Group({self._name}, {fields!r})"

    def to_dict(self) -> dict:
        return dict(self._config._groups.get(self._name, {}))

    def set_bounds(self, field: str, min: Any = None, max: Any = None, **extra: Any) -> "Config":
        return self._config.set_bounds(self._name, field, min=min, max=max, **extra)

    def set_values(self, field: str, values: list) -> "Config":
        return self._config.set_values(self._name, field, values)
