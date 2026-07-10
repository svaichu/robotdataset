"""Two-way Optuna compatibility for `Config`.

Building a search space and running a study::

    distributions = cfg.to_optuna_distributions()   # optuna.distributions, keyed 'group.field'

    def objective(trial):
        trial_cfg = cfg.suggest(trial)               # sweepable fields replaced with trial suggestions
        return train(trial_cfg)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

Loading the winning config back onto the schema::

    best_cfg = cfg.from_optuna_study(study)           # study.best_params
    best_cfg = cfg.from_optuna_params(trial.params)    # any dotted-key params dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .field import FieldSpec

if TYPE_CHECKING:
    from .config import Config


def _require_optuna():
    """Import optuna lazily so the rest of the config system needs no extra deps."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna support requires optuna. Install it with "
            "'pip install robotdataset[optuna]' or 'pip install optuna'."
        ) from exc
    return optuna


def _bounds_or_raise(spec: FieldSpec) -> dict:
    bounds = spec.bounds or {}
    if "min" not in bounds or "max" not in bounds:
        raise ValueError(
            f"Field '{spec.name}' is not a valid Optuna range: bounds must set both "
            f"'min' and 'max', got {bounds!r}"
        )
    return bounds


def field_to_distribution(spec: FieldSpec):
    """Convert a sweepable `FieldSpec` into an `optuna.distributions` object.

    `bounds` may carry `log` (bool) and `step` on top of `min`/`max`, matching
    the extra kwargs `Config.set_bounds` already passes through.
    """
    optuna = _require_optuna()
    if spec.values is not None:
        return optuna.distributions.CategoricalDistribution(list(spec.values))

    bounds = _bounds_or_raise(spec)
    log = bool(bounds.get("log", False))
    step = bounds.get("step")
    if spec.type == "int":
        return optuna.distributions.IntDistribution(
            int(bounds["min"]), int(bounds["max"]), log=log, step=int(step) if step is not None else 1
        )
    return optuna.distributions.FloatDistribution(float(bounds["min"]), float(bounds["max"]), log=log, step=step)


def suggest_field(trial: Any, key: str, spec: FieldSpec) -> Any:
    """Ask an `optuna.Trial` to suggest a value for one sweepable field."""
    if spec.values is not None:
        return trial.suggest_categorical(key, list(spec.values))

    bounds = _bounds_or_raise(spec)
    log = bool(bounds.get("log", False))
    step = bounds.get("step")
    if spec.type == "int":
        kwargs: dict = {"log": log}
        if step is not None:
            kwargs["step"] = int(step)
        return trial.suggest_int(key, int(bounds["min"]), int(bounds["max"]), **kwargs)

    kwargs = {"log": log}
    if step is not None:
        kwargs["step"] = step
    return trial.suggest_float(key, float(bounds["min"]), float(bounds["max"]), **kwargs)


def to_optuna_distributions(config: "Config", groups: Optional[list[str]] = None) -> dict:
    """Build a dict of `optuna.distributions`, keyed `'group.field'`, from sweepable fields.

    Suitable for `optuna.study.Study.enqueue_trial` / `add_trial` / distribution-aware
    samplers that need the search space up front.
    """
    _require_optuna()
    distributions: dict = {}
    for group, fields in config._schema.items():
        if groups is not None and group not in groups:
            continue
        for field, spec in fields.items():
            if spec.is_sweepable():
                distributions[f"{group}.{field}"] = field_to_distribution(spec)
    return distributions


def suggest(config: "Config", trial: Any, groups: Optional[list[str]] = None) -> "Config":
    """Return a new `Config` with every sweepable field replaced by `trial`'s suggestion.

    Non-sweepable fields keep their current values unchanged. Use inside an
    Optuna objective function::

        def objective(trial):
            trial_cfg = cfg.suggest(trial)
            return train(trial_cfg)
    """
    _require_optuna()
    cfg = config.clone()
    for group, fields in config._schema.items():
        if groups is not None and group not in groups:
            continue
        for field, spec in fields.items():
            if spec.is_sweepable():
                key = f"{group}.{field}"
                cfg._set_field_strict(group, field, suggest_field(trial, key, spec))
    return cfg


def from_optuna_params(config: "Config", params: dict) -> "Config":
    """Return a new `Config` with dotted `'group.field'` params (e.g. `trial.params`) applied."""
    cfg = config.clone()
    for key, value in params.items():
        if "." not in key:
            raise ValueError(f"Optuna param key must be 'group.field', got {key!r}")
        group, _, field = key.partition(".")
        cfg._set_field_strict(group, field, value)
    return cfg


def from_optuna_study(config: "Config", study: Any) -> "Config":
    """Return a new `Config` with the winning params from a completed `optuna.Study` applied."""
    return from_optuna_params(config, study.best_params)
