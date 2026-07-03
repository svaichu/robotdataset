"""CLI entrypoint: convert a Config YAML/JSON file into a W&B sweep config.

Usage:
    python -m robotdataset.configuration_system.main config.yaml sweep.yaml
"""

from __future__ import annotations

import argparse

from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Input YAML/JSON config file")
    parser.add_argument("sweep_path", help="Output W&B sweep YAML file")
    parser.add_argument("--method", default="bayes", choices=["bayes", "grid", "random"])
    args = parser.parse_args()

    cfg = Config.from_file(args.config_path)
    cfg.to_sweep_file(args.sweep_path, method=args.method)
    print(f"Wrote sweep config to {args.sweep_path}")


if __name__ == "__main__":
    main()
