"""Unit tests for qnc.hardware (Stage 16 Qiskit translation helpers).

Covers z_from_counts (pure function, no hardware needed) and the measured
circuit-building path via a local Aer/Qiskit statevector-backend simulation
(qnc/hardware.py's `build_circuit(measure=True)`), satisfying CLAUDE.md
invariant 1 without touching real hardware.
"""

import numpy as np
from qiskit_aer import AerSimulator

from qnc.hardware import build_circuit, z_from_counts


def test_z_from_counts_all_zero_gives_z_plus_one():
    counts = {"000": 1024}
    z, z_err = z_from_counts(counts, shots=1024, num_classes=3)
    assert np.allclose(z, [1.0, 1.0, 1.0])
    assert np.allclose(z_err, [0.0, 0.0, 0.0])


def test_z_from_counts_all_one_gives_z_minus_one():
    counts = {"111": 1024}
    z, z_err = z_from_counts(counts, shots=1024, num_classes=3)
    assert np.allclose(z, [-1.0, -1.0, -1.0])
    assert np.allclose(z_err, [0.0, 0.0, 0.0])


def test_z_from_counts_half_half_gives_z_zero_with_nonzero_error():
    counts = {"000": 512, "111": 512}
    z, z_err = z_from_counts(counts, shots=1024, num_classes=3)
    assert np.allclose(z, [0.0, 0.0, 0.0])
    expected_err = 2.0 * np.sqrt(0.5 * 0.5 / 1024)
    assert np.allclose(z_err, expected_err)


def test_z_from_counts_qubit_0_is_rightmost_character():
    counts = {"001": 1024}
    z, _ = z_from_counts(counts, shots=1024, num_classes=3)
    assert np.allclose(z, [-1.0, 1.0, 1.0])


def test_build_circuit_measure_true_matches_statevector_probabilities():
    n_qubits, num_classes = 4, 3
    rng = np.random.default_rng(0)
    x = rng.uniform(-np.pi, np.pi, size=n_qubits)
    weights = rng.uniform(0, 2 * np.pi, size=(2, n_qubits, 3))

    qc = build_circuit(x, weights, n_qubits, num_classes, measure=True)
    sim = AerSimulator()
    result = sim.run(qc, shots=20000, seed_simulator=0).result()
    counts = result.get_counts()

    z_hw, _ = z_from_counts(counts, shots=20000, num_classes=num_classes)

    from qnc.hardware import statevector_z
    z_exact = statevector_z(x, weights, n_qubits, num_classes)

    assert np.max(np.abs(z_hw - z_exact)) < 0.02
