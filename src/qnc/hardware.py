"""Qiskit translation of the trained VQC for hardware inference (Stage 16).

Implements SPEC3_ADDENDUM.md section 22: re-express the R0, reupload=True
StronglyEntanglingLayers circuit (src/qnc/models.py VQC._circuit_body) in
Qiskit for execution on IBM hardware. No training happens here -- weights
are loaded from a stored checkpoint.

Entangler range: PennyLane's StronglyEntanglingLayers default range formula
is `ranges[l] = (l % (n_wires - 1)) + 1`, but VQC._circuit_body (reupload
path) calls StronglyEntanglingLayers separately per layer with a length-1
weight slice, so the range formula is evaluated fresh each call on
n_layers=1 -> ranges=[1] every time. The entangler is therefore CNOT ring
with range=1 on every layer, not the varying range a longer single call
would produce. Verified against qml.StronglyEntanglingLayers.
compute_decomposition source and pinned by
tests/test_stage16_translation_gate.py.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector


def build_circuit(
    x: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    num_classes: int,
    measure: bool = False,
) -> QuantumCircuit:
    """Builds the data-reuploading VQC circuit for one sample.

    Args:
        x: angle-encoded input, shape (n_qubits,), already scaled to
            [-pi, pi] (output of qnc.data.pca_encode).
        weights: main-circuit StronglyEntanglingLayers weights, shape
            (n_layers, n_qubits, 3), as stored in the VQC checkpoint's
            `weights` parameter.
        measure: if True, adds Z-basis measurement of the first
            `num_classes` qubits into a classical register of that size
            (hardware/shot execution). If False, no measurement (statevector
            validation).
    """
    n_layers = weights.shape[0]
    qc = QuantumCircuit(n_qubits, num_classes if measure else 0)

    for l in range(n_layers):
        for j in range(n_qubits):
            qc.ry(float(x[j]), j)
        for q in range(n_qubits):
            phi, theta, omega = (float(weights[l, q, 0]), float(weights[l, q, 1]), float(weights[l, q, 2]))
            qc.rz(phi, q)
            qc.ry(theta, q)
            qc.rz(omega, q)
        if n_qubits > 1:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)

    if measure:
        for c in range(num_classes):
            qc.measure(c, c)

    return qc


def statevector_z(x: np.ndarray, weights: np.ndarray, n_qubits: int, num_classes: int) -> np.ndarray:
    """Exact z_i via Aer/Qiskit statevector (no shots). Used only for the
    local translation-gate validation against the PennyLane pipeline."""
    qc = build_circuit(x, weights, n_qubits, num_classes, measure=False)
    sv = Statevector.from_instruction(qc)
    z = np.zeros(num_classes)
    for c in range(num_classes):
        z[c] = sv.expectation_value(Pauli("Z"), qargs=[c]).real
    return z


def z_from_counts(counts: dict, shots: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """z_i and binomial shot-noise std error from a hardware/simulator counts
    dict (Qiskit bitstring keys, qubit 0 = rightmost character).

    z_i[c] = P(qubit c == 0) - P(qubit c == 1) = 1 - 2*P(qubit c == 1).
    Binomial std on P(qubit c == 1) is sqrt(p(1-p)/shots); propagated to z
    via the factor 2 (SPEC3_ADDENDUM.md section 22, "binomial propagation").
    """
    z = np.zeros(num_classes)
    z_err = np.zeros(num_classes)
    for c in range(num_classes):
        ones = 0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            if bits[-(c + 1)] == "1":
                ones += count
        p1 = ones / shots
        z[c] = 1.0 - 2.0 * p1
        z_err[c] = 2.0 * np.sqrt(max(p1 * (1.0 - p1), 0.0) / shots)
    return z, z_err
