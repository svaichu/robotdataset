"""CLI entrypoint: convert a Config YAML/JSON file into a W&B sweep config.

Any field from the config file can be overridden on the command line as a
dotted `--<group>.<field>` option before the sweep is exported.

Usage:
    python -m robotdataset.configuration_system.main config.yaml sweep.yaml \\
        --method bayes --training.num_epochs 200
"""

from __future__ import annotations

import argparse

from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Input YAML/JSON config file")
    parser.add_argument("sweep_path", help="Output W&B sweep YAML file")
    parser.add_argument("--method", default="bayes", choices=["bayes", "grid", "random"])
    known, _ = parser.parse_known_args()

    cfg = Config.from_file(known.config_path)
    cfg.add_arguments(parser)
    args = parser.parse_args()
    cfg.apply_args(args)

    cfg.to_sweep_file(args.sweep_path, method=args.method)
    print(f"Wrote sweep config to {args.sweep_path}")


if __name__ == "__main__":
    main()
