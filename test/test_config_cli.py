import argparse

import pytest

from robotdataset.configuration_system import Config

SAMPLE_YAML = """
dataset:
  name: oxe
  batch_size: 32
  shuffle: true
  cameras: [wrist, front]

training:
  learning_rate: 1.0e-4
  optimizer: adam
  num_epochs: 100
  extra:
    warmup: 10
"""


def make_cfg(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(SAMPLE_YAML)
    return Config.from_yaml(path), path


def test_parse_args_overrides_with_learned_types(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--training.learning_rate", "1e-3", "--dataset.batch_size", "64"])

    assert cfg.training.learning_rate == 1e-3
    assert isinstance(cfg.training.learning_rate, float)
    assert cfg.dataset.batch_size == 64
    assert isinstance(cfg.dataset.batch_size, int)


def test_unpassed_args_keep_file_values(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--training.learning_rate", "1e-3"])

    assert cfg.training.optimizer == "adam"
    assert cfg.training.num_epochs == 100
    assert cfg.dataset.name == "oxe"


def test_wandb_agent_equals_style(tmp_path):
    # `wandb agent` invokes: python train.py --training.learning_rate=0.001
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--training.learning_rate=0.001", "--dataset.shuffle=false"])

    assert cfg.training.learning_rate == 0.001
    assert cfg.dataset.shuffle is False


def test_bool_bare_flag_and_explicit_values(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.dataset.shuffle = False
    cfg.parse_args(["--dataset.shuffle"])
    assert cfg.dataset.shuffle is True

    cfg.parse_args(["--dataset.shuffle", "no"])
    assert cfg.dataset.shuffle is False

    cfg.parse_args(["--dataset.shuffle", "True"])
    assert cfg.dataset.shuffle is True


def test_list_accepts_json_and_comma_separated(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--dataset.cameras", '["top", "side"]'])
    assert cfg.dataset.cameras == ["top", "side"]

    cfg.parse_args(["--dataset.cameras", "wrist,front,top"])
    assert cfg.dataset.cameras == ["wrist", "front", "top"]

    cfg.parse_args(["--dataset.cameras", "1,2,3"])
    assert cfg.dataset.cameras == [1, 2, 3]  # elements are JSON-parsed


def test_dict_parses_json_object(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--training.extra", '{"warmup": 20, "decay": 0.9}'])
    assert cfg.training.extra == {"warmup": 20, "decay": 0.9}


def test_values_become_argparse_choices(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.set_values("training", "optimizer", ["adam", "sgd"])

    cfg.parse_args(["--training.optimizer", "sgd"])
    assert cfg.training.optimizer == "sgd"

    with pytest.raises(SystemExit):
        cfg.parse_args(["--training.optimizer", "rmsprop"])


def test_unknown_override_errors(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    with pytest.raises(SystemExit):
        cfg.parse_args(["--training.missing_field", "1"])


def test_parse_args_non_strict_ignores_unknown(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--training.learning_rate", "1e-3", "--verbose"], strict=False)
    assert cfg.training.learning_rate == 1e-3


def test_add_arguments_composes_with_existing_parser(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="run0")
    cfg.add_arguments(parser)

    args = parser.parse_args(["--run-name", "exp1", "--training.num_epochs", "5"])
    cfg.apply_args(args)  # non-dotted keys like run_name are ignored

    assert args.run_name == "exp1"
    assert cfg.training.num_epochs == 5


def test_add_arguments_groups_filter(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    parser = argparse.ArgumentParser()
    cfg.add_arguments(parser, groups=["training"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset.batch_size", "8"])


def test_apply_args_accepts_string_dict_like_wandb_config(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.apply_args({"training.learning_rate": "3e-4", "dataset.shuffle": "false"})

    assert cfg.training.learning_rate == 3e-4
    assert cfg.dataset.shuffle is False


def test_apply_args_unknown_dotted_key_raises(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    with pytest.raises(KeyError, match="Unknown field 'training.missing'"):
        cfg.apply_args({"training.missing": 1})


def test_from_cli_loads_config_and_applies_overrides(tmp_path):
    _, path = make_cfg(tmp_path)
    cfg = Config.from_cli(
        ["--config", str(path), "--training.learning_rate", "5e-4", "--dataset.name", "libero"]
    )

    assert cfg.training.learning_rate == 5e-4
    assert cfg.dataset.name == "libero"
    assert cfg.training.num_epochs == 100  # untouched file value


def test_from_cli_uses_default_config(tmp_path):
    _, path = make_cfg(tmp_path)
    cfg = Config.from_cli(["--dataset.batch_size", "128"], default_config=path)
    assert cfg.dataset.batch_size == 128


def test_from_cli_explicit_config_beats_default(tmp_path):
    _, path = make_cfg(tmp_path)
    other = tmp_path / "other.yaml"
    other.write_text("dataset:\n  name: libero\n")

    cfg = Config.from_cli(["--config", str(other)], default_config=path)
    assert cfg.dataset.name == "libero"
    assert "batch_size" not in cfg.dataset


def test_from_cli_requires_a_config(tmp_path):
    with pytest.raises(SystemExit):
        Config.from_cli([])


def test_schema_defaults_survive_cli_overrides(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    cfg.parse_args(["--training.learning_rate", "1e-3"])

    spec = cfg.schema("training", "learning_rate")
    assert spec.default == 1e-4  # file default preserved; only the value changes
    assert cfg.training.learning_rate == 1e-3
