"""Sweep runner (Stage 4).

Invoked as `python -m qnc.sweep configs/sweeps/<name>.yaml`. Drives run.py's
dispatch_train across a list of config overrides defined in a sweep YAML:

    name: ablation_entangling
    base_config: configs/stage2_vqc_c2_diagnostic.yaml
    seeds: [0, 1, 2]
    variants:
      - label: entangling_true
        overrides: {}
      - label: entangling_false
        overrides: {model.entangling: false}

Runs sequentially (one variant x seed at a time, per PLAYBOOK.md Stage 4).
Each run's config gets `experiment` set to "<sweep_name>_<variant_label>" so
results/ directory names stay readable; a manifest.json mapping
{variant_label: {seed: run_id}} is written since run_ids also embed
timestamps and can't otherwise be recovered from the sweep config alone.
"""

import copy
import json
from datetime import datetime
from pathlib import Path

import yaml

from qnc.figures import make_figures, make_sweep_comparison_figure
from qnc.report import build_cross_config_table
from qnc.run import dispatch_train


def _set_by_dotted_path(config: dict, path: str, value) -> None:
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def merge_overrides(base_config: dict, overrides: dict) -> dict:
    """Deep-copies base_config and applies each {"a.b.c": value} override."""
    merged = copy.deepcopy(base_config)
    for path, value in overrides.items():
        _set_by_dotted_path(merged, path, value)
    return merged


def run_sweep(sweep_config: dict, base_config: dict, results_dir: str | Path = "results") -> tuple[dict, Path]:
    """Drives every variant x seed combination, writes manifest.json.

    Returns (manifest, manifest_path). manifest is {label: {str(seed): run_id}}.
    """
    name = sweep_config["name"]
    seeds = sweep_config["seeds"]
    manifest: dict[str, dict[str, str]] = {}

    for variant in sweep_config["variants"]:
        label = variant["label"]
        overrides = variant.get("overrides", {})
        manifest[label] = {}
        for seed in seeds:
            merged = merge_overrides(base_config, overrides)
            merged["experiment"] = f"{name}_{label}"
            merged["seed"] = seed
            run_id = dispatch_train(merged, seed_override=seed)
            make_figures(run_id, results_dir=results_dir)
            manifest[label][str(seed)] = run_id
            print(f"[sweep] {label} seed={seed} done: {run_id}", flush=True)
        print(f"[sweep] variant {label} complete ({len(manifest[label])} seeds)", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = Path(results_dir) / "sweeps" / f"{name}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    make_sweep_comparison_figure(sweep_config, manifest, sweep_dir / "comparison.png", results_dir=results_dir)
    report = build_cross_config_table(manifest, results_dir=results_dir)
    with open(sweep_dir / "report.md", "w") as f:
        f.write(report)

    return manifest, manifest_path


def main() -> None:
    """CLI: ``python -m qnc.sweep configs/sweeps/<name>.yaml``."""
    import argparse

    parser = argparse.ArgumentParser(description="Drive a sweep of config overrides sequentially.")
    parser.add_argument("sweep_config", type=str, help="Path to a sweep YAML file")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)
    with open(sweep_config["base_config"]) as f:
        base_config = yaml.safe_load(f)

    manifest, manifest_path = run_sweep(sweep_config, base_config, results_dir=args.results_dir)
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
