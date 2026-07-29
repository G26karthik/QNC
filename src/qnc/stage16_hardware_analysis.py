"""Stage 16 Task C: hardware-vs-simulator z-space analysis.

Reads the raw hardware counts saved by the Stage 16 hardware job
(results/stage16_hardware/<job_id>/counts.json + job_log.json), recomputes
z_i on hardware (Z-basis counts, binomial shot-noise error bars) and on the
exact simulator (same 180 blobs3 samples, same trained parameters), and
produces the SPEC3_ADDENDUM.md section 22 comparison: table + z-space
class-mean overlay figure + NC1_z bar figure. No training; inference-only,
pure post-processing of already-collected data.
"""

import json
from pathlib import Path

import numpy as np
import torch

from qnc.data import load_blobs3, pca_encode
from qnc.figures import make_hardware_nc1z_bar_figure, make_hardware_zspace_comparison_figure
from qnc.hardware import statevector_z, z_from_counts
from qnc.metrics import d_box, margins, nc1_z_metrics
from qnc.models import VQC

N_QUBITS, NUM_CLASSES, N_LAYERS, SEED = 4, 3, 8, 0
CKPT_PATH = "results/transition_L_L8_s0_20260712-194651/checkpoints/epoch_0600.pt"


def _load_reference_data_and_model():
    x_train_raw, y_train, x_test_raw, y_test = load_blobs3(
        samples_per_class=60, blob_dist=3.0, blob_std=0.8, test_fraction=0.2,
        seed=SEED, ambient_dim=N_QUBITS,
    )
    x_train, x_test = pca_encode(x_train_raw, x_test_raw, n_qubits=N_QUBITS, seed=SEED)
    x_all = torch.cat([x_train, x_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)

    model = VQC(
        n_qubits=N_QUBITS, num_classes=NUM_CLASSES, n_layers=N_LAYERS,
        reupload=True, entangling=True, encoding="angle", readout_family="R0",
    )
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()
    return x_all.numpy(), y_all.numpy(), model.weights.detach().numpy()


def compute_hw_and_sim_z(
    job_id: str, shots: int = 2048, results_dir: str | Path = "results",
    weights_np: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recomputes z_hw (+ shot-noise err) and z_sim for a saved Stage 16
    hardware job. `weights_np` defaults to the trained anchor checkpoint;
    pass untrained weights (see Stage 16b HW3) to recompute against those
    instead."""
    hw_dir = Path(results_dir) / "stage16_hardware" / job_id
    with open(hw_dir / "counts.json") as f:
        counts_list = json.load(f)

    x_all, y_all, ckpt_weights_np = _load_reference_data_and_model()
    if weights_np is None:
        weights_np = ckpt_weights_np
    n = x_all.shape[0]
    assert len(counts_list) == n, f"counts ({len(counts_list)}) / samples ({n}) mismatch"

    z_hw = np.zeros((n, NUM_CLASSES))
    z_hw_err = np.zeros((n, NUM_CLASSES))
    z_sim = np.zeros((n, NUM_CLASSES))
    for i in range(n):
        z_hw[i], z_hw_err[i] = z_from_counts(counts_list[i], shots=shots, num_classes=NUM_CLASSES)
        z_sim[i] = statevector_z(x_all[i], weights_np, N_QUBITS, NUM_CLASSES)
    return z_hw, z_hw_err, z_sim, y_all


def run_analysis(job_id: str, shots: int = 2048, results_dir: str | Path = "results") -> dict:
    hw_dir = Path(results_dir) / "stage16_hardware" / job_id
    with open(hw_dir / "job_log.json") as f:
        job_log = json.load(f)

    z_hw, z_hw_err, z_sim, y_all = compute_hw_and_sim_z(job_id, shots, results_dir)
    n = z_hw.shape[0]

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

    fig_dir = hw_dir / "figures"
    make_hardware_zspace_comparison_figure(
        nc1_hw["zbar_c"], _class_mean_err(z_hw_err, y_all), nc1_sim["zbar_c"],
        fig_dir / "zspace_hw_vs_sim.png",
    )
    make_hardware_nc1z_bar_figure(nc1_hw["nc1_z"], nc1_sim["nc1_z"], fig_dir / "nc1z_hw_vs_sim.png")

    summary = {
        "job_id": job_id,
        "backend": job_log.get("backend"),
        "n_samples": n,
        "shots_per_sample": shots,
        "nc1_z_hardware": nc1_hw["nc1_z"],
        "nc1_z_simulator": nc1_sim["nc1_z"],
        "r_z_hardware": nc1_hw["r_z"],
        "r_z_simulator": nc1_sim["r_z"],
        "d_box_hardware": dbox_hw,
        "d_box_simulator": dbox_sim,
        "mean_margin_hardware": margins_hw["mean_margin"],
        "mean_margin_simulator": margins_sim["mean_margin"],
        "min_margin_hardware": margins_hw["min_margin"],
        "min_margin_simulator": margins_sim["min_margin"],
        "zbar_c_hardware": nc1_hw["zbar_c"].tolist(),
        "zbar_c_simulator": nc1_sim["zbar_c"].tolist(),
        "max_abs_diff_z": max_abs_diff,
        "mean_abs_diff_z": mean_abs_diff,
        "mean_shot_noise_std_z": mean_z_err,
        "fraction_within_shot_noise": within_shot_noise_frac,
    }
    with open(hw_dir / "stage16_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _class_mean_err(z_err: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Standard error of each class mean from independent per-sample
    binomial shot-noise std (SPEC3_ADDENDUM.md section 22, error propagation
    for the mean of independent estimates)."""
    classes = np.unique(labels)
    out = np.zeros((len(classes), z_err.shape[1]))
    for idx, c in enumerate(classes):
        errs_c = z_err[labels == c]
        out[idx] = np.sqrt(np.sum(errs_c**2, axis=0)) / errs_c.shape[0]
    return out


if __name__ == "__main__":
    import sys

    result = run_analysis(sys.argv[1])
    print(json.dumps(result, indent=2))
