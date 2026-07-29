"""Stage 16b: contraction decomposition (HW1/HW2, existing counts) +
untrained-parameter hardware control (HW3, one new hardware job's counts).

Implements PREDICTIONS.md's Stage 16b preregistration. No training; all
functions here are pure post-processing of already-collected hardware
counts (Stage 16's trained-parameter job and Stage 16b's untrained-parameter
job) plus the exact simulator recomputation.
"""

import json
from pathlib import Path

import numpy as np
import torch

from qnc.figures import make_hardware_zspace_rescaled_figure
from qnc.metrics import equiangle_dev_z, nc1_z_metrics
from qnc.models import VQC
from qnc.stage16_hardware_analysis import _class_mean_err, compute_hw_and_sim_z

N_QUBITS, NUM_CLASSES, N_LAYERS, SEED = 4, 3, 8, 0


def untrained_weights() -> np.ndarray:
    """Reconstructs the anchor's seed-0 UNTRAINED weights (PREDICTIONS.md
    Stage 16b HW3): `train.py` calls `torch.manual_seed(seed)` immediately
    before constructing the VQC, and nothing in between touches the global
    torch RNG (data loading / PCA use independently seeded generators), so
    this reproduces the exact pre-training initial weights."""
    torch.manual_seed(SEED)
    model = VQC(
        n_qubits=N_QUBITS, num_classes=NUM_CLASSES, n_layers=N_LAYERS,
        reupload=True, entangling=True, encoding="angle", readout_family="R0",
    )
    return model.weights.detach().numpy()


def _sigma_w_b(z: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sigma_W^z, Sigma_B^z (C x C matrices, SPEC3_ADDENDUM.md section 18.1
    formulas), returned directly rather than just their traces/NC1_z."""
    classes = np.unique(labels)
    num_classes = len(classes)
    num_samples, dim = z.shape

    sigma_w = np.zeros((dim, dim))
    class_means = np.zeros((num_classes, dim))
    for idx, c in enumerate(classes):
        zc = z[labels == c]
        mu_c = zc.mean(axis=0)
        class_means[idx] = mu_c
        diffs = zc - mu_c
        sigma_w += diffs.T @ diffs
    sigma_w /= num_samples

    mu_g = z.mean(axis=0)
    sigma_b = np.zeros((dim, dim))
    for idx in range(num_classes):
        d = (class_means[idx] - mu_g).reshape(-1, 1)
        sigma_b += d @ d.T
    sigma_b /= num_classes
    return sigma_w, sigma_b


def fit_hw1(z_hw: np.ndarray, z_sim: np.ndarray, y_all: np.ndarray, nc1_z_hw_observed: float) -> dict:
    """HW1: least-squares z_hw = lambda*z_sim + residual (through origin,
    pooled over all 540 (sample,component) pairs), then predicts nc1_z_hw
    from lambda^2-scaled simulator Sigma_W/B plus the empirical residual's
    own Sigma_W/B."""
    z_hw_flat = z_hw.flatten()
    z_sim_flat = z_sim.flatten()

    lam = float(np.sum(z_hw_flat * z_sim_flat) / np.sum(z_sim_flat**2))
    residual = z_hw - lam * z_sim
    resid_flat = residual.flatten()

    ss_res = float(np.sum(resid_flat**2))
    ss_tot = float(np.sum((z_hw_flat - z_hw_flat.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    residual_var = float(np.var(resid_flat))

    sigma_w_sim, sigma_b_sim = _sigma_w_b(z_sim, y_all)
    sigma_w_resid, sigma_b_resid = _sigma_w_b(residual, y_all)

    sigma_w_pred = lam**2 * sigma_w_sim + sigma_w_resid
    sigma_b_pred = lam**2 * sigma_b_sim + sigma_b_resid
    nc1_z_pred = float((1.0 / NUM_CLASSES) * np.trace(sigma_w_pred @ np.linalg.pinv(sigma_b_pred)))

    rel_err = abs(nc1_z_pred - nc1_z_hw_observed) / nc1_z_hw_observed

    if lam - 1e-9 <= 0.85 and lam >= 0.6 - 1e-9 and r2 >= 0.9 and rel_err <= 0.15:
        verdict = "CONFIRMED"
    elif (0.6 <= lam <= 0.85 or r2 >= 0.9) and rel_err <= 0.30:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    return {
        "lambda": lam, "r2": r2, "residual_var": residual_var,
        "nc1_z_predicted": nc1_z_pred, "nc1_z_observed": nc1_z_hw_observed,
        "relative_error": rel_err, "verdict": verdict,
    }


def bootstrap_hw2(
    z_hw: np.ndarray, z_hw_err: np.ndarray, z_sim: np.ndarray, y_all: np.ndarray,
    n_resamples: int = 1000, seed: int = 0,
) -> dict:
    """HW2: bootstrap equiangle_dev_z over shot noise (>=1000 resamples),
    compared to the simulator's exact (noiseless) value."""
    rng = np.random.default_rng(seed)
    boot_vals = np.zeros(n_resamples)
    for b in range(n_resamples):
        noise = rng.normal(0.0, z_hw_err)
        boot_vals[b] = equiangle_dev_z(z_hw + noise, y_all)

    hw_mean, hw_std = float(boot_vals.mean()), float(boot_vals.std())
    ci95 = (float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5)))
    ci99_7 = (float(np.percentile(boot_vals, 0.15)), float(np.percentile(boot_vals, 99.85)))
    sim_val = equiangle_dev_z(z_sim, y_all)

    if ci95[0] <= sim_val <= ci95[1]:
        verdict = "CONFIRMED"
    elif ci99_7[0] <= sim_val <= ci99_7[1]:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    return {
        "equiangle_dev_z_hardware_mean": hw_mean, "equiangle_dev_z_hardware_std": hw_std,
        "equiangle_dev_z_hardware_ci95": ci95, "equiangle_dev_z_hardware_ci99_7": ci99_7,
        "equiangle_dev_z_simulator": sim_val, "n_resamples": n_resamples, "verdict": verdict,
    }


def score_hw3(nc1_z_untrained: float, nc1_z_trained: float = 0.6748530085791519) -> dict:
    ratio = nc1_z_untrained / nc1_z_trained
    if ratio >= 10:
        verdict = "CONFIRMED"
    elif ratio >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"
    return {"nc1_z_untrained": nc1_z_untrained, "nc1_z_trained": nc1_z_trained, "ratio": ratio, "verdict": verdict}


def run_hw1_hw2(job_id: str, shots: int = 2048, results_dir: str | Path = "results") -> dict:
    hw_dir = Path(results_dir) / "stage16_hardware" / job_id
    z_hw, z_hw_err, z_sim, y_all = compute_hw_and_sim_z(job_id, shots, results_dir)
    nc1_hw = nc1_z_metrics(z_hw, y_all)

    hw1 = fit_hw1(z_hw, z_sim, y_all, nc1_hw["nc1_z"])
    hw2 = bootstrap_hw2(z_hw, z_hw_err, z_sim, y_all)

    zbar_c_hw_rescaled = nc1_hw["zbar_c"] / hw1["lambda"]
    nc1_sim = nc1_z_metrics(z_sim, y_all)
    fig_dir = hw_dir / "figures"
    make_hardware_zspace_rescaled_figure(
        zbar_c_hw_rescaled, _class_mean_err(z_hw_err, y_all) / hw1["lambda"], nc1_sim["zbar_c"],
        fig_dir / "zspace_hw_rescaled_vs_sim.png",
    )

    result = {"job_id": job_id, "hw1": hw1, "hw2": hw2}
    with open(hw_dir / "stage16b_hw1_hw2_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_hw3(untrained_job_id: str, shots: int = 2048, results_dir: str | Path = "results") -> dict:
    hw_dir = Path(results_dir) / "stage16_hardware" / untrained_job_id
    with open(hw_dir / "job_log.json") as f:
        job_log = json.load(f)

    weights_untrained = untrained_weights()
    z_hw, z_hw_err, z_sim, y_all = compute_hw_and_sim_z(
        untrained_job_id, shots, results_dir, weights_np=weights_untrained,
    )
    nc1_hw = nc1_z_metrics(z_hw, y_all)
    nc1_sim = nc1_z_metrics(z_sim, y_all)
    hw3 = score_hw3(nc1_hw["nc1_z"])

    result = {
        "job_id": untrained_job_id, "backend": job_log.get("backend"),
        "nc1_z_untrained_hardware": nc1_hw["nc1_z"], "nc1_z_untrained_simulator": nc1_sim["nc1_z"],
        "hw3": hw3,
    }
    with open(hw_dir / "stage16b_hw3_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
