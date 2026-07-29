"""Stage 16 Task A: translation gate.

Re-expresses the trained n=4 L=8 R0 anchor circuit (transition_L L8, seed 0,
reupload=true, final checkpoint) in Qiskit and validates it against the
existing PennyLane pipeline on all 180 blobs3 samples. Hard gate per
SPEC3_ADDENDUM.md section 22 / PLAYBOOK_REMAINING.md Stage 16: no hardware
submission is permitted until this test is green.
"""

import numpy as np
import torch

from qnc.data import load_blobs3, pca_encode
from qnc.hardware import statevector_z
from qnc.metrics import z_features
from qnc.models import VQC

CKPT_PATH = "results/transition_L_L8_s0_20260712-194651/checkpoints/epoch_0600.pt"
N_QUBITS = 4
NUM_CLASSES = 3
N_LAYERS = 8
SEED = 0


def _load_anchor_samples_and_model():
    x_train_raw, y_train, x_test_raw, y_test = load_blobs3(
        samples_per_class=60, blob_dist=3.0, blob_std=0.8, test_fraction=0.2,
        seed=SEED, ambient_dim=N_QUBITS,
    )
    x_train, x_test = pca_encode(x_train_raw, x_test_raw, n_qubits=N_QUBITS, seed=SEED)
    x_all = torch.cat([x_train, x_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)

    model = VQC(
        n_qubits=N_QUBITS,
        num_classes=NUM_CLASSES,
        n_layers=N_LAYERS,
        reupload=True,
        entangling=True,
        beta_0=5.0,
        beta_learnable=True,
        encoding="angle",
        readout_family="R0",
    )
    state_dict = torch.load(CKPT_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return x_all, y_all, model


def test_translation_gate_matches_pennylane_within_1e6():
    x_all, y_all, model = _load_anchor_samples_and_model()
    assert x_all.shape[0] == 180, f"expected 180 blobs3 samples, got {x_all.shape[0]}"

    with torch.no_grad():
        states = model.state(x_all).detach().cpu().numpy()
    rhos = np.einsum("ni,nj->nij", states, states.conj())
    z_ref = z_features(rhos, n_qubits=N_QUBITS, num_classes=NUM_CLASSES)

    weights_np = model.weights.detach().cpu().numpy()
    x_np = x_all.detach().cpu().numpy()

    z_qiskit = np.zeros_like(z_ref)
    for i in range(x_all.shape[0]):
        z_qiskit[i] = statevector_z(x_np[i], weights_np, N_QUBITS, NUM_CLASSES)

    max_abs_diff = np.max(np.abs(z_ref - z_qiskit))
    assert max_abs_diff < 1e-6, f"translation gate FAILED: max abs diff {max_abs_diff} >= 1e-6"
