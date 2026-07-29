"""Stage 18 B5: hw-vs-sim z-space analysis for the ibm_kingston replication
job. Same protocol as qnc.stage16_hardware_analysis.run_analysis, pointed
at results/stage18_hardware/ instead of results/stage16_hardware/."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from qnc.hardware import z_from_counts
from qnc.metrics import d_box, margins, nc1_z_metrics
from qnc.stage16_hardware_analysis import _load_reference_data_and_model

JOB_ID = "d9egj2hhtsac739e7ts0"
HW_DIR = Path("results/stage18_hardware") / JOB_ID
NUM_CLASSES = 3
SHOTS = 2048


def main():
    with open(HW_DIR / "counts.json") as f:
        counts_list = json.load(f)

    x_all, y_all, weights_np = _load_reference_data_and_model()
    from qnc.hardware import statevector_z

    n = x_all.shape[0]
    assert len(counts_list) == n

    z_hw = np.zeros((n, NUM_CLASSES))
    z_hw_err = np.zeros((n, NUM_CLASSES))
    z_sim = np.zeros((n, NUM_CLASSES))
    for i in range(n):
        z_hw[i], z_hw_err[i] = z_from_counts(counts_list[i], shots=SHOTS, num_classes=NUM_CLASSES)
        z_sim[i] = statevector_z(x_all[i], weights_np, 4, NUM_CLASSES)

    nc1_hw = nc1_z_metrics(z_hw, y_all)
    nc1_sim = nc1_z_metrics(z_sim, y_all)
    margins_hw = margins(z_hw, y_all)
    margins_sim = margins(z_sim, y_all)
    dbox_hw = d_box(nc1_hw["zbar_c"])
    dbox_sim = d_box(nc1_sim["zbar_c"])

    max_abs_diff = float(np.max(np.abs(z_hw - z_sim)))
    mean_abs_diff = float(np.mean(np.abs(z_hw - z_sim)))
    mean_z_err = float(np.mean(z_hw_err))
    within_shot_noise_frac = float(np.mean(np.abs(z_hw - z_sim) <= z_hw_err))

    summary = {
        "job_id": JOB_ID,
        "backend": "ibm_kingston",
        "n_samples": n,
        "shots_per_sample": SHOTS,
        "nc1_z_hardware": nc1_hw["nc1_z"],
        "nc1_z_simulator": nc1_sim["nc1_z"],
        "r_z_hardware": nc1_hw["r_z"],
        "r_z_simulator": nc1_sim["r_z"],
        "mean_margin_hardware": margins_hw["mean_margin"],
        "mean_margin_simulator": margins_sim["mean_margin"],
        "min_margin_hardware": margins_hw["min_margin"],
        "min_margin_simulator": margins_sim["min_margin"],
        "d_box_hardware": dbox_hw,
        "d_box_simulator": dbox_sim,
        "max_abs_diff_z": max_abs_diff,
        "mean_abs_diff_z": mean_abs_diff,
        "mean_shot_noise_std_z": mean_z_err,
        "fraction_within_shot_noise": within_shot_noise_frac,
    }
    with open(HW_DIR / "stage18_b5_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
