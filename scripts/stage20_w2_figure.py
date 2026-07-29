"""Stage 20 Arm W2 figure driver: classical MLP tr(Sigma_W) vs epoch beside
matched VQC (no-reupload) total_var vs epoch. Classical side reads freshly
rerun metrics.jsonl directly (within_class_scatter_trace is logged per
epoch already); VQC side reuses Stage 19 Arm V2(b) checkpoints via the same
_zspace_checkpoint_walk used by stage19_score.py -- no VQC rerun.
"""
import json
import sys

sys.path.insert(0, "src")
import yaml

from qnc.figures import make_stage20_w2_contraction_vs_rotation_figure
from qnc.reanalyze import _zspace_checkpoint_walk
from qnc.sweep import merge_overrides

CLASSICAL_RUNS = {
    "MNIST (classical)": [
        "stage19_classical_mnist_pca4_default_s0_20260725-173002",
        "stage19_classical_mnist_pca4_default_s1_20260725-173036",
        "stage19_classical_mnist_pca4_default_s2_20260725-173144",
    ],
    "Fashion-MNIST (classical)": [
        "stage19_classical_fmnist_pca4_default_s0_20260725-173237",
        "stage19_classical_fmnist_pca4_default_s1_20260725-173338",
        "stage19_classical_fmnist_pca4_default_s2_20260725-173521",
    ],
}

MNIST_BASE = "configs/stage19_mnist_base.yaml"
FMNIST_BASE = "configs/stage19_fmnist_base.yaml"
REUPLOAD_FALSE_OV = {"model.layers": 16, "model.reupload": False}

VQC_RUNS = {
    "MNIST (VQC no-reupload)": {
        "base": MNIST_BASE,
        "runs": {
            0: "stage19_mnist_campaign_false_reupload_false_s0_20260725-134240",
            1: "stage19_mnist_campaign_false_reupload_false_s1_20260725-143055",
            2: "stage19_mnist_campaign_false_reupload_false_s2_20260725-150115",
            3: "stage19_mnist_campaign_false_reupload_false_s3_20260725-152752",
            4: "stage19_mnist_campaign_false_reupload_false_s4_20260725-154815",
        },
    },
    "Fashion-MNIST (VQC no-reupload)": {
        "base": FMNIST_BASE,
        "runs": {
            0: "stage19_fmnist_campaign_false_reupload_false_s0_20260725-134120",
            1: "stage19_fmnist_campaign_false_reupload_false_s1_20260725-142929",
            2: "stage19_fmnist_campaign_false_reupload_false_s2_20260725-145945",
            3: "stage19_fmnist_campaign_false_reupload_false_s3_20260725-152642",
            4: "stage19_fmnist_campaign_false_reupload_false_s4_20260725-154713",
        },
    },
}


def load_classical_records(run_id: str) -> list[dict]:
    lines = [json.loads(l) for l in open(f"results/{run_id}/metrics.jsonl").read().strip().split("\n")]
    return [r for r in lines if r["split"] == "train"]


def load_vqc_records(base_path: str, seed: int, run_id: str) -> list[dict]:
    base_config = yaml.safe_load(open(base_path))
    config = merge_overrides(base_config, REUPLOAD_FALSE_OV)
    return _zspace_checkpoint_walk(config, seed, run_id)


if __name__ == "__main__":
    classical_records = {
        label: [load_classical_records(rid) for rid in run_ids] for label, run_ids in CLASSICAL_RUNS.items()
    }
    vqc_records = {
        label: [load_vqc_records(spec["base"], seed, rid) for seed, rid in sorted(spec["runs"].items())]
        for label, spec in VQC_RUNS.items()
    }
    make_stage20_w2_contraction_vs_rotation_figure(classical_records, vqc_records)
    print("wrote figs/fig19_contraction_vs_rotation.png")

    # Summary stats for STAGE20_FINDINGS.md
    for label, runs in classical_records.items():
        ratios = [runs_i[-1]["within_class_scatter_trace"] / runs_i[0]["within_class_scatter_trace"] for runs_i in runs]
        print(f"{label}: tr(Sigma_W) final/init ratios = {[round(r, 3) for r in ratios]}")
    for label, runs in vqc_records.items():
        rels = [max(abs(r["total_var"] - runs_i[0]["total_var"]) / abs(runs_i[0]["total_var"]) for r in runs_i) for runs_i in runs]
        print(f"{label}: total_var max relative change = {[f'{r:.2e}' for r in rels]}")
