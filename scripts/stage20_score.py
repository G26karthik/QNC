"""Stage 20 scoring (Arms W1, W4): total_var invariance, NC1_z ratio,
fiber_fraction per run off checkpoints, plus TPT/test-acc from metrics.jsonl
-- same score_run pattern as stage19_score.py, but reads overrides straight
from the sweep YAML + manifest.json instead of hardcoding run_ids (Stage 20's
sweeps run to completion rather than being killed mid-flight, so a manifest
always exists).

Usage: python stage20_score.py <sweep_yaml_path> <manifest_json_path>
"""
import json
import sys

sys.path.insert(0, "src")
import yaml

from qnc.reanalyze import _zspace_checkpoint_walk
from qnc.sweep import merge_overrides


def score_run(base_config: dict, overrides: dict, seed: int, run_id: str) -> dict:
    config = merge_overrides(base_config, overrides)
    records = _zspace_checkpoint_walk(config, seed, run_id)
    tv0 = records[0]["total_var"]
    tv_max_rel = max(abs(r["total_var"] - tv0) / abs(tv0) for r in records)
    nc1_0 = records[0]["nc1_z"]
    nc1_final = records[-1]["nc1_z"]
    fiber_final = records[-1]["fiber_fraction"]

    metrics_path = f"results/{run_id}/metrics.jsonl"
    lines = [json.loads(l) for l in open(metrics_path).read().strip().split("\n")]
    test_lines = [r for r in lines if r.get("split") == "test"]
    final_test = test_lines[-1]

    return {
        "run_id": run_id,
        "total_var_max_rel_change": tv_max_rel,
        "nc1_z_ratio": nc1_final / nc1_0,
        "fiber_fraction_final": fiber_final,
        "n_checkpoints": len(records),
        "test_acc_final": final_test["acc"],
        "tpt_reached": bool(final_test["tpt_reached"]),
        "e0_epoch": final_test["e0_epoch"],
    }


def score_sweep(sweep_yaml_path: str, manifest_path: str) -> dict:
    sweep_config = yaml.safe_load(open(sweep_yaml_path))
    base_config = yaml.safe_load(open(sweep_config["base_config"]))
    manifest = json.load(open(manifest_path))

    overrides_by_label = {v["label"]: v.get("overrides", {}) for v in sweep_config["variants"]}

    results = {}
    for label, seeds in manifest.items():
        results[label] = {}
        for seed_str, run_id in sorted(seeds.items(), key=lambda kv: int(kv[0])):
            results[label][seed_str] = score_run(base_config, overrides_by_label[label], int(seed_str), run_id)
    return results


if __name__ == "__main__":
    sweep_yaml_path, manifest_path = sys.argv[1], sys.argv[2]
    results = score_sweep(sweep_yaml_path, manifest_path)
    out_name = sweep_yaml_path.split("/")[-1].replace(".yaml", "_score.json")
    with open(out_name, "w") as f:
        json.dump(results, f, indent=2)
    for label, seeds in results.items():
        tpt_count = sum(1 for r in seeds.values() if r["tpt_reached"])
        print(f"{label}: {tpt_count}/{len(seeds)} TPT")
    print(f"wrote {out_name}")
