"""Stage 18 arm B4: noisy-simulation inference (PREDICTIONS.md "Stage 18
preregistration" section, arm B4).

Restructured (per the preregistration) as an inference-only arm mirroring
Stage 16/16b's hardware protocol exactly: a per-gate depolarizing channel
is applied to the SAME already-trained n=4, L=8, R0, reupload=true, seed-0
anchor checkpoint used by the hardware arm
(`results/transition_L_L8_s0_20260712-194651/checkpoints/epoch_0600.pt`),
and z is measured over the same 180 blobs3 samples (144 train + 36 test).
Exact (shots=None) density-matrix simulation -- no training, no sampling
noise; "seeds" do not apply here (see PREDICTIONS.md B4).
"""

import numpy as np
import pennylane as qml

from qnc.models import VQC
from qnc.stage16_hardware_analysis import _load_reference_data_and_model


def _noisy_expval_qnode(model: VQC, p: float):
    """Builds a `default.mixed` QNode reproducing `model`'s own
    `_circuit_body` (so the circuit definition is never duplicated), with
    `qml.DepolarizingChannel(p)` inserted after every gate via
    `qml.insert(..., position="all")`. R0 readout only (raw Z_c
    expectations, pre-beta, matching `qnc.hardware.statevector_z`'s
    convention)."""
    dev = qml.device("default.mixed", wires=model.n_qubits)

    @qml.insert(op=qml.DepolarizingChannel, op_args=p, position="all")
    @qml.qnode(dev)
    def circuit(x, weights):
        model._circuit_body(x, weights)
        return [qml.expval(qml.PauliZ(c)) for c in range(model.num_classes)]

    return circuit


def noisy_z(x_all: np.ndarray, weights_np: np.ndarray, model: VQC, p: float) -> np.ndarray:
    """z_i, shape (n, num_classes), under an exact (shots=None)
    density-matrix simulation with a depolarizing channel of strength `p`
    inserted after every gate. `p=0` must reproduce the noiseless
    statevector result exactly -- the translation gate verified by
    `tests/test_stage18_noisy_translation_gate.py`, mirroring Stage 16's
    `tests/test_stage16_translation_gate.py`."""
    circuit = _noisy_expval_qnode(model, p)
    n = x_all.shape[0]
    z = np.zeros((n, model.num_classes))
    for i in range(n):
        z[i] = np.asarray(circuit(x_all[i], weights_np))
    return z


def load_anchor_data_and_model():
    """Re-exports the Stage 16 anchor loader so this module has a single,
    documented entry point (same 180 samples, same trained checkpoint)."""
    return _load_reference_data_and_model()
