import optuna
import pytest

from robotdataset.configuration_system import Config


def _make_cfg() -> Config:
    cfg = Config()
    cfg.define("training", "learning_rate", default=1e-4, type="float")
    cfg.define("training", "optimizer", default="adam", type="str")
    cfg.define("training", "num_epochs", default=100, type="int")
    cfg.training(
        learning_rate=1e-4,
        optimizer="adam",
        num_epochs=100,
    )
    cfg.set_bounds("training", "learning_rate", min=1e-5, max=1e-2, log=True)
    cfg.set_values("training", "optimizer", ["adam", "sgd"])
    return cfg


def test_to_optuna_distributions_maps_bounds_and_values():
    cfg = _make_cfg()
    distributions = cfg.to_optuna_distributions()

    lr_dist = distributions["training.learning_rate"]
    assert isinstance(lr_dist, optuna.distributions.FloatDistribution)
    assert lr_dist.low == 1e-5
    assert lr_dist.high == 1e-2
    assert lr_dist.log is True

    opt_dist = distributions["training.optimizer"]
    assert isinstance(opt_dist, optuna.distributions.CategoricalDistribution)
    assert opt_dist.choices == ("adam", "sgd")

    # Fixed (non-sweepable) fields aren't part of the search space.
    assert "training.num_epochs" not in distributions


def test_suggest_returns_new_config_without_mutating_original():
    cfg = _make_cfg()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()

    trial_cfg = cfg.suggest(trial)

    assert 1e-5 <= trial_cfg.training.learning_rate <= 1e-2
    assert trial_cfg.training.optimizer in ("adam", "sgd")
    assert trial_cfg.training.num_epochs == 100  # fixed field carried over unchanged

    # Original config untouched.
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.optimizer == "adam"


def test_suggest_is_usable_inside_an_optuna_study():
    cfg = _make_cfg()

    def objective(trial):
        trial_cfg = cfg.suggest(trial)
        return (trial_cfg.training.learning_rate - 1e-3) ** 2

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=5)

    assert len(study.trials) == 5
    for trial in study.trials:
        assert set(trial.params.keys()) == {"training.learning_rate", "training.optimizer"}


def test_from_optuna_params_applies_dotted_keys_onto_a_clone():
    cfg = _make_cfg()
    best_cfg = cfg.from_optuna_params({"training.learning_rate": 3e-3, "training.optimizer": "sgd"})

    assert best_cfg.training.learning_rate == 3e-3
    assert best_cfg.training.optimizer == "sgd"
    assert best_cfg.training.num_epochs == 100

    # Original config untouched.
    assert cfg.training.learning_rate == 1e-4


def test_from_optuna_study_round_trips_the_winning_config():
    cfg = _make_cfg()

    def objective(trial):
        trial_cfg = cfg.suggest(trial)
        return (trial_cfg.training.learning_rate - 1e-3) ** 2

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=10)

    best_cfg = cfg.from_optuna_study(study)

    assert best_cfg.training.learning_rate == study.best_params["training.learning_rate"]
    assert best_cfg.training.optimizer == study.best_params["training.optimizer"]
    assert best_cfg.training.num_epochs == 100


def test_from_optuna_params_raises_for_non_dotted_key():
    cfg = _make_cfg()
    with pytest.raises(ValueError, match="group.field"):
        cfg.from_optuna_params({"learning_rate": 1e-3})


def test_int_field_with_bounds_uses_suggest_int():
    cfg = Config()
    cfg.define("training", "batch_size", default=32, type="int")
    cfg.training(batch_size=32)
    cfg.set_bounds("training", "batch_size", min=8, max=256, step=8)

    distributions = cfg.to_optuna_distributions()
    assert isinstance(distributions["training.batch_size"], optuna.distributions.IntDistribution)

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    trial_cfg = cfg.suggest(trial)
    assert isinstance(trial_cfg.training.batch_size, int)
    assert 8 <= trial_cfg.training.batch_size <= 256
