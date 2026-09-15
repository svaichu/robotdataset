"""Example train script: build a Config's schema with add_argument(), no YAML file.

add_argument() mirrors argparse.ArgumentParser.add_argument but takes a
dotted "group.field" name, so it registers the field and exposes it as a
CLI option in the same call.

    python example/train_cli_programmatic_example.py
    python example/train_cli_programmatic_example.py --training.learning_rate 1e-3 --training.optimizer sgd
    python example/train_cli_programmatic_example.py --help
"""

from robotdataset import Config


def build_config() -> Config:
    cfg = Config(description="Example VLA training run")
    cfg.add_argument("dataset.name", default="oxe")
    cfg.add_argument("dataset.batch_size", default=32, type=int)
    cfg.add_argument("training.learning_rate", default=1e-4, type=float)
    cfg.add_argument("training.optimizer", default="adam", choices=["adam", "sgd"])
    cfg.add_argument("training.num_epochs", default=100, type=int)
    return cfg


def main() -> None:
    cfg = build_config()
    cfg.parse_args()
    print(cfg)


if __name__ == "__main__":
    main()
