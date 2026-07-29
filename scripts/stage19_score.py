"""Stage 19 scoring: total_var invariance, NC1_z ratio, fiber_fraction per
run, off checkpoints -- same pattern as stage18_b3_score.py.
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
    return {
        "run_id": run_id,
        "total_var_max_rel_change": tv_max_rel,
        "nc1_z_ratio": nc1_final / nc1_0,
        "fiber_fraction_final": fiber_final,
        "n_checkpoints": len(records),
    }


# Stage 19's original consolidated sweeps (stage19_mnist_campaign,
# stage19_fmnist_campaign, stage19_r3_mnist, stage19_r3_fmnist) were killed
# mid-flight by the background-task infrastructure and never wrote a
# manifest.json (qnc.sweep only writes one after ALL variant x seed combos
# finish). The missing seeds were recovered via separate "_remainder"/
# "_false" sweeps launched later. This maps each dataset/variant/seed to
# its actual completed run_id (all verified to reach epoch 600), rather
# than relying on any single manifest.

MNIST_BASE = "configs/stage19_mnist_base.yaml"
FMNIST_BASE = "configs/stage19_fmnist_base.yaml"

REUPLOAD_TRUE_OV = {"model.layers": 16, "model.reupload": True}
REUPLOAD_FALSE_OV = {"model.layers": 16, "model.reupload": False}
R3_OV = {"model.layers": 16, "model.reupload": True, "model.readout_family": "R3", "model.measurement_layers": 2}

RUN_SPECS = {
    "stage19_mnist_campaign": {
        "base_config": MNIST_BASE,
        "variants": {
            "reupload_true": {
                "overrides": REUPLOAD_TRUE_OV,
                "runs": {
                    0: "stage19_mnist_campaign_reupload_true_s0_20260724-211115",
                    1: "stage19_mnist_campaign_remainder_reupload_true_s1_20260725-133928",
                    2: "stage19_mnist_campaign_remainder_reupload_true_s2_20260725-143051",
                    3: "stage19_mnist_campaign_remainder_reupload_true_s3_20260725-150612",
                    4: "stage19_mnist_campaign_remainder_reupload_true_s4_20260725-153613",
                },
            },
            "reupload_false": {
                "overrides": REUPLOAD_FALSE_OV,
                "runs": {
                    0: "stage19_mnist_campaign_false_reupload_false_s0_20260725-134240",
                    1: "stage19_mnist_campaign_false_reupload_false_s1_20260725-143055",
                    2: "stage19_mnist_campaign_false_reupload_false_s2_20260725-150115",
                    3: "stage19_mnist_campaign_false_reupload_false_s3_20260725-152752",
                    4: "stage19_mnist_campaign_false_reupload_false_s4_20260725-154815",
                },
            },
        },
    },
    "stage19_fmnist_campaign": {
        "base_config": FMNIST_BASE,
        "variants": {
            "reupload_true": {
                "overrides": REUPLOAD_TRUE_OV,
                "runs": {
                    0: "stage19_fmnist_campaign_reupload_true_s0_20260724-204221",
                    1: "stage19_fmnist_campaign_reupload_true_s1_20260724-211656",
                    2: "stage19_fmnist_campaign_remainder_reupload_true_s2_20260725-134245",
                    3: "stage19_fmnist_campaign_remainder_reupload_true_s3_20260725-143556",
                    4: "stage19_fmnist_campaign_remainder_reupload_true_s4_20260725-151116",
                },
            },
            "reupload_false": {
                "overrides": REUPLOAD_FALSE_OV,
                "runs": {
                    0: "stage19_fmnist_campaign_false_reupload_false_s0_20260725-134120",
                    1: "stage19_fmnist_campaign_false_reupload_false_s1_20260725-142929",
                    2: "stage19_fmnist_campaign_false_reupload_false_s2_20260725-145945",
                    3: "stage19_fmnist_campaign_false_reupload_false_s3_20260725-152642",
                    4: "stage19_fmnist_campaign_false_reupload_false_s4_20260725-154713",
                },
            },
        },
    },
    "stage19_r3_mnist": {
        "base_config": MNIST_BASE,
        "variants": {
            "r3_l16": {
                "overrides": R3_OV,
                "runs": {
                    0: "stage19_r3_mnist_r3_l16_s0_20260724-211120",
                    1: "stage19_r3_mnist_remainder_r3_l16_s1_20260725-134154",
                    2: "stage19_r3_mnist_remainder_r3_l16_s2_20260725-143843",
                },
            },
        },
    },
    "stage19_r3_fmnist": {
        "base_config": FMNIST_BASE,
        "variants": {
            "r3_l16": {
                "overrides": R3_OV,
                "runs": {
                    0: "stage19_r3_fmnist_r3_l16_s0_20260724-204307",
                    1: "stage19_r3_fmnist_r3_l16_s1_20260724-211946",
                    2: "stage19_r3_fmnist_remainder_r3_l16_s2_20260725-124236",
                },
            },
        },
    },
}


if __name__ == "__main__":
    all_results = {}
    for name, spec in RUN_SPECS.items():
        base = yaml.safe_load(open(spec["base_config"]))
        all_results[name] = {}
        for label, variant in spec["variants"].items():
            all_results[name][label] = {}
            for seed, run_id in sorted(variant["runs"].items()):
                all_results[name][label][str(seed)] = score_run(base, variant["overrides"], seed, run_id)
        print(f"scored {name}: {list(spec['variants'].keys())}, seeds {sorted(next(iter(spec['variants'].values()))['runs'])}")
    with open("stage19_score.json", "w") as f:
        json.dump(all_results, f, indent=2)
