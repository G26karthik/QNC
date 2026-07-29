"""Single-run entry point.

Invoked as `python -m qnc.run configs/<name>.yaml` per CLAUDE.md invariant 2.
Loads a YAML config, dispatches to train.py, and writes results/<run_id>/.
"""

import argparse

import yaml

from qnc.train import train, train_vqc


def dispatch_train(config: dict, seed_override: int | None = None) -> str:
    """Routes a loaded config to train() or train_vqc() by config["model"]["type"].

    Shared by run.py (single run) and sweep.py (Stage 4, many runs).
    """
    model_type = config["model"]["type"]
    if model_type == "vqc":
        return train_vqc(config, seed_override=seed_override)
    elif model_type == "mlp":
        return train(config, seed_override=seed_override)
    else:
        raise ValueError(f"unknown model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_id = dispatch_train(config, seed_override=args.seed)
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
