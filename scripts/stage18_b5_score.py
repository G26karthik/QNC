"""Stage 18 B5 scoring: lambda_kingston contraction fit (HW1 methodology)
+ equiangle_dev_z bootstrap CI (HW2 methodology), reused unchanged from
qnc.stage16b_analysis against the ibm_kingston job's z_hw/z_sim."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from qnc.hardware import statevector_z, z_from_counts
from qnc.metrics import nc1_z_metrics
from qnc.stage16_hardware_analysis import _load_reference_data_and_model
from qnc.stage16b_analysis import bootstrap_hw2, fit_hw1

JOB_ID = "d9egj2hhtsac739e7ts0"
HW_DIR = Path("results/stage18_hardware") / JOB_ID
NUM_CLASSES = 3
SHOTS = 2048


def main():
    with open(HW_DIR / "counts.json") as f:
        counts_list = json.load(f)
    x_all, y_all, weights_np = _load_reference_data_and_model()
    n = x_all.shape[0]

    z_hw = np.zeros((n, NUM_CLASSES))
    z_hw_err = np.zeros((n, NUM_CLASSES))
    z_sim = np.zeros((n, NUM_CLASSES))
    for i in range(n):
        z_hw[i], z_hw_err[i] = z_from_counts(counts_list[i], shots=SHOTS, num_classes=NUM_CLASSES)
        z_sim[i] = statevector_z(x_all[i], weights_np, 4, NUM_CLASSES)

    nc1_hw = nc1_z_metrics(z_hw, y_all)
    hw1 = fit_hw1(z_hw, z_sim, y_all, nc1_hw["nc1_z"])
    hw2 = bootstrap_hw2(z_hw, z_hw_err, z_sim, y_all)

    lam = hw1["lambda"]
    lam_in_range = 0.55 <= lam <= 0.85
    angle_holds = hw2["verdict"] in ("CONFIRMED",)

    if lam_in_range and angle_holds:
        verdict = "CONFIRMED"
    elif lam_in_range or angle_holds:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    result = {"job_id": JOB_ID, "backend": "ibm_kingston", "hw1": hw1, "hw2": hw2,
              "lambda_in_0.55_0.85": lam_in_range, "angle_survival_holds": angle_holds,
              "b5_verdict": verdict}
    with open(HW_DIR / "stage18_b5_score.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
