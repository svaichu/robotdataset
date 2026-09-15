"""Example train script: load a YAML config and override it from the CLI.

Every field learned from the config file is automatically exposed as a
`--<group>.<field>` option -- no extra wiring needed.

    python example/train_cli_example.py
    python example/train_cli_example.py --training.learning_rate 3e-4 --dataset.batch_size 128
    python example/train_cli_example.py --config other_config.yaml --training.optimizer sgd
    python example/train_cli_example.py --help
"""

from pathlib import Path

from robotdataset import Config

DEFAULT_CONFIG = Path(__file__).parent / "sample_config.yaml"


def main() -> None:
    cfg = Config.from_cli(default_config=DEFAULT_CONFIG, description="Example VLA training run")
    print(cfg)


if __name__ == "__main__":
    main()
