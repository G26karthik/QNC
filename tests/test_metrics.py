"""SPEC.md §6 unit tests for src/qnc/metrics.py. Exact expected values, no
loosened tolerances (CLAUDE.md invariant 1)."""

import numpy as np
import pennylane as qml
import pytest
import torch

from qnc.metrics import (
    trace_distance,
    hs_inner,
    hs_norm,
    class_means,
    centered_means,
    qnc1_metrics,
    qnc1_fisher,
    qnc1_ratio,
    overlap_offdiag,
    purity,
    qnc2_metrics,
    qfi_matrix,
    qfi_rank,
    reduced_density,
    von_neumann_entropy,
    classical_nc1,
    classical_nc2,
    within_class_scatter_trace,
    pauli_z_operator,
    nc1_z_metrics,
    margins,
    d_box,
    equiangle_dev_z,
    measured_subspace_basis,
    measured_subspace_basis_r1,
    measured_subspace_basis_r2,
    measured_subspace_basis_r3,
    fiber_decomposition,
    loss_visible_subspace_decomposition,
    isotropic_fiber_null,
    total_within_class_hs_variance,
)

# --- shared fixtures: single-qubit density matrices ---

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ket_plus = (ket0 + ket1) / np.sqrt(2)

rho0 = np.outer(ket0, ket0.conj())
rho1 = np.outer(ket1, ket1.conj())
rho_plus = np.outer(ket_plus, ket_plus.conj())

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def random_pure_state(dim, rng):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    return v


class TestTraceDistance:
    def test_identical_states_zero(self):
        assert trace_distance(rho0, rho0) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_states_one(self):
        assert trace_distance(rho0, rho1) == pytest.approx(1.0, abs=1e-10)

    def test_zero_plus_states(self):
        assert trace_distance(rho0, rho_plus) == pytest.approx(
            1 / np.sqrt(2), abs=1e-8
        )

    def test_pure_state_identity_on_random_pairs(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            psi = random_pure_state(4, rng)
            phi = random_pure_state(4, rng)
            rho = np.outer(psi, psi.conj())
            sigma = np.outer(phi, phi.conj())
            overlap = np.abs(np.vdot(psi, phi)) ** 2
            expected = np.sqrt(1 - overlap)
            assert trace_distance(rho, sigma) == pytest.approx(expected, abs=1e-8)


class TestHilbertSchmidt:
    def test_hs_inner_orthogonal_paulis(self):
        val = hs_inner(PAULI_X, PAULI_Z)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_hs_norm_pauli_x(self):
        assert hs_norm(PAULI_X) == pytest.approx(np.sqrt(2), abs=1e-10)


class TestClassMeans:
    def test_class_means_and_global_mean(self):
        # 2 samples of rho0 in class 0, 2 samples of rho1 in class 1
        rhos = np.stack([rho0, rho0, rho1, rho1])
        labels = np.array([0, 0, 1, 1])
        means, global_mean = class_means(rhos, labels)
        assert means.shape == (2, 2, 2)
        np.testing.assert_allclose(means[0], rho0, atol=1e-12)
        np.testing.assert_allclose(means[1], rho1, atol=1e-12)
        expected_global = 0.5 * rho0 + 0.5 * rho1
        np.testing.assert_allclose(global_mean, expected_global, atol=1e-12)

    def test_centered_means_traceless_hermitian(self):
        rng = np.random.default_rng(1)
        rhos = []
        labels = []
        for c in range(3):
            for _ in range(5):
                psi = random_pure_state(4, rng)
                rhos.append(np.outer(psi, psi.conj()))
                labels.append(c)
        rhos = np.stack(rhos)
        labels = np.array(labels)
        means, global_mean = class_means(rhos, labels)
        m = centered_means(means, global_mean)
        for c in range(3):
            assert np.trace(m[c]).real == pytest.approx(0.0, abs=1e-10)
            assert abs(np.trace(m[c]).imag) < 1e-10
            np.testing.assert_allclose(m[c], m[c].conj().T, atol=1e-10)


class TestQNC1Metrics:
    def test_zero_when_all_samples_equal_class_mean(self):
        rhos = np.stack([rho0, rho0, rho1, rho1])
        labels = np.array([0, 0, 1, 1])
        means, global_mean = class_means(rhos, labels)
        result = qnc1_metrics(rhos, labels, means, global_mean)
        assert result["qnc1_trace"] == pytest.approx(0.0, abs=1e-10)
        assert result["qnc1_hs"] == pytest.approx(0.0, abs=1e-10)

    def test_rel_is_nan_when_denominator_below_guard(self):
        # both classes share the same rho -> global mean == class means -> M_c == 0
        rhos = np.stack([rho0, rho0, rho0, rho0])
        labels = np.array([0, 0, 1, 1])
        means, global_mean = class_means(rhos, labels)
        result = qnc1_metrics(rhos, labels, means, global_mean)
        assert np.isnan(result["qnc1_rel"])


class TestQNC2Metrics:
    def test_hand_built_c3_simplex_etf_in_r2(self):
        # Embed R^2 as span{X, Z} (orthogonal, ||X||_HS = ||Z||_HS = sqrt(2)).
        # A simplex ETF of 3 unit vectors in R^2 sits at 120 degrees apart.
        angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        means = np.stack(
            [np.cos(t) * PAULI_X + np.sin(t) * PAULI_Z for t in angles]
        )
        global_mean = np.zeros((2, 2), dtype=complex)
        result = qnc2_metrics(means, global_mean)

        assert result["equiangle_dev"] == pytest.approx(0.0, abs=1e-10)
        assert result["equinorm_cv"] == pytest.approx(0.0, abs=1e-10)
        C = 3
        for i in range(C):
            for j in range(C):
                if i != j:
                    assert result["cos"][i, j] == pytest.approx(-0.5, abs=1e-10)


class TestEntanglementEntropy:
    def test_product_state_zero_entropy(self):
        ket00 = np.kron(ket0, ket0)
        rho00 = np.outer(ket00, ket00.conj())
        rho_a = reduced_density(rho00, n_qubits=2, num_keep=1)
        assert von_neumann_entropy(rho_a) == pytest.approx(0.0, abs=1e-10)

    def test_bell_state_entropy_is_one_bit(self):
        bell = (np.kron(ket0, ket0) + np.kron(ket1, ket1)) / np.sqrt(2)
        rho_bell = np.outer(bell, bell.conj())
        rho_a = reduced_density(rho_bell, n_qubits=2, num_keep=1)
        assert von_neumann_entropy(rho_a) == pytest.approx(1.0, abs=1e-10)

    def test_ghz4_half_partition_entropy_is_one_bit(self):
        ket0000 = np.kron(np.kron(np.kron(ket0, ket0), ket0), ket0)
        ket1111 = np.kron(np.kron(np.kron(ket1, ket1), ket1), ket1)
        ghz4 = (ket0000 + ket1111) / np.sqrt(2)
        rho_ghz4 = np.outer(ghz4, ghz4.conj())
        rho_a = reduced_density(rho_ghz4, n_qubits=4, num_keep=2)
        assert von_neumann_entropy(rho_a) == pytest.approx(1.0, abs=1e-10)


class TestClassicalNC:
    def test_nc1_synthetic_analytic_value(self):
        # 1D features, 2 classes, N_c=2 each.
        # class 0: {-1, 1} -> mu_0=0; class 1: {3, 5} -> mu_1=4.
        # Sigma_W = (1/N) sum_c sum_i (h_i-mu_c)^2 = (2+2)/4 = 1.0
        # mu_G = mean(-1,1,3,5) = 2; Sigma_B = (1/C) sum_c (mu_c-mu_G)^2
        #      = ((0-2)^2 + (4-2)^2) / 2 = 4.0
        # NC1 = (1/C) * tr(Sigma_W @ pinv(Sigma_B)) = (1/2) * (1.0/4.0) = 0.125
        h = np.array([[-1.0], [1.0], [3.0], [5.0]])
        labels = np.array([0, 0, 1, 1])
        assert classical_nc1(h, labels) == pytest.approx(0.125, abs=1e-6)

    def test_within_class_scatter_trace_matches_manual_sigma_w(self):
        # Same synthetic setup as test_nc1_synthetic_analytic_value:
        # Sigma_W = (1/N) sum_c sum_i (h_i-mu_c)^2 = (2+2)/4 = 1.0, and since
        # dim=1 here, tr(Sigma_W) == Sigma_W == 1.0 exactly (SPEC3_ADDENDUM
        # section 24).
        h = np.array([[-1.0], [1.0], [3.0], [5.0]])
        labels = np.array([0, 0, 1, 1])
        assert within_class_scatter_trace(h, labels) == pytest.approx(1.0, abs=1e-10)

    def test_within_class_scatter_trace_multivariate_matches_manual_computation(self):
        rng = np.random.default_rng(0)
        h = rng.normal(size=(40, 5))
        labels = np.array([0] * 20 + [1] * 20)
        sigma_w = np.zeros((5, 5))
        for c in (0, 1):
            hc = h[labels == c]
            diffs = hc - hc.mean(axis=0)
            sigma_w += diffs.T @ diffs
        sigma_w /= h.shape[0]
        expected = float(np.trace(sigma_w))
        assert within_class_scatter_trace(h, labels) == pytest.approx(expected, abs=1e-10)
        assert within_class_scatter_trace(h, labels) >= 0.0

    def test_nc2_hand_built_c3_simplex_etf(self):
        angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        h = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        labels = np.array([0, 1, 2])
        result = classical_nc2(h, labels)

        assert result["equiangle_dev"] == pytest.approx(0.0, abs=1e-10)
        assert result["equinorm_cv"] == pytest.approx(0.0, abs=1e-10)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert result["cos"][i, j] == pytest.approx(-0.5, abs=1e-10)


class TestSPEC2NormalizedWitnesses:
    """SPEC2_ADDENDUM.md §11.4 required unit tests."""

    def test_qnc1_fisher_zero_when_all_samples_equal_class_mean(self):
        rhos = np.stack([rho0, rho0, rho1, rho1])
        labels = np.array([0, 0, 1, 1])
        means, global_mean = class_means(rhos, labels)
        qnc1 = qnc1_metrics(rhos, labels, means, global_mean)
        qnc2 = qnc2_metrics(means, global_mean)
        assert qnc1_fisher(qnc1["qnc1_hs"], qnc2["class_mean_norms"]) == pytest.approx(0.0, abs=1e-10)

    def test_qnc1_ratio_zero_when_all_samples_equal_class_mean(self):
        rhos = np.stack([rho0, rho0, rho1, rho1])
        labels = np.array([0, 0, 1, 1])
        means, global_mean = class_means(rhos, labels)
        qnc1 = qnc1_metrics(rhos, labels, means, global_mean)
        assert qnc1_ratio(qnc1["qnc1_trace"], means) == pytest.approx(0.0, abs=1e-10)

    def test_overlap_offdiag_orthogonal_means_zero(self):
        means = np.stack([rho0, rho1])
        assert overlap_offdiag(means) == pytest.approx(0.0, abs=1e-10)

    def test_overlap_offdiag_identical_pure_means_one(self):
        means = np.stack([rho_plus, rho_plus])
        assert overlap_offdiag(means) == pytest.approx(1.0, abs=1e-10)

    def test_purity_of_pure_state_is_one(self):
        assert purity(rho0) == pytest.approx(1.0, abs=1e-10)

    def test_purity_of_maximally_mixed_two_qubits_is_quarter(self):
        rho = np.eye(4, dtype=complex) / 4
        assert purity(rho) == pytest.approx(0.25, abs=1e-10)


class TestQFI:
    """SPEC2_ADDENDUM.md §12.2 (empirical transition locator) and §16
    (autograd-vs-parameter-shift agreement) required tests."""

    def test_ry_one_qubit_qfi_is_quarter(self):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(theta):
            qml.RY(theta[0], wires=0)
            return qml.state()

        theta = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        psi_fn = lambda: circuit(theta).unsqueeze(0)
        g = qfi_matrix(psi_fn, theta)
        assert g.shape == (1, 1)
        assert g[0, 0] == pytest.approx(0.25, abs=1e-8)

    def test_redundant_parameters_rank_one(self):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(theta):
            qml.RY(theta[0], wires=0)
            qml.RY(theta[1], wires=0)
            return qml.state()

        theta = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
        psi_fn = lambda: circuit(theta).unsqueeze(0)
        g = qfi_matrix(psi_fn, theta)
        assert qfi_rank(g) == 1

    def test_autograd_agrees_with_parameter_shift_metric_tensor(self):
        # metric_tensor's exact (Hadamard-test) mode needs one free auxiliary
        # wire beyond the 2 the circuit itself uses.
        dev = qml.device("default.qubit", wires=3)
        theta_values = [0.3, 0.5, 0.2]
        theta_np = qml.numpy.array(theta_values, requires_grad=True)

        def circuit_body(theta):
            qml.RY(theta[0], wires=0)
            qml.RY(theta[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(theta[2], wires=0)

        @qml.qnode(dev)
        def circuit_ps(theta):
            circuit_body(theta)
            return qml.state()

        mt = qml.metric_tensor(circuit_ps, approx=None)(theta_np)

        dev_torch = qml.device("default.qubit", wires=2)

        @qml.qnode(dev_torch, interface="torch", diff_method="backprop")
        def circuit_torch(theta):
            circuit_body(theta)
            return qml.state()

        theta_t = torch.tensor(theta_values, dtype=torch.float64, requires_grad=True)
        psi_fn = lambda: circuit_torch(theta_t).unsqueeze(0)
        g = qfi_matrix(psi_fn, theta_t)
        np.testing.assert_allclose(g, np.array(mt), atol=1e-6)


class TestSPEC3ZSpaceWitnesses:
    """SPEC3_ADDENDUM.md §18.5 required unit tests."""

    def test_pauli_z_orthogonality_n4(self):
        # <Z_j, Z_k>_HS = delta_jk * 2^n, n=4.
        ops = [pauli_z_operator(4, q) for q in range(4)]
        for j in range(4):
            for k in range(4):
                expected = 16.0 if j == k else 0.0
                assert hs_inner(ops[j], ops[k]) == pytest.approx(expected, abs=1e-10)

    def test_nc1_z_zero_when_z_equals_class_mean(self):
        # z_i == zbar_c per class, distinct class means -> NC1_z = 0, r_z = 0.
        z = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]])
        labels = np.array([0, 0, 1, 1])
        result = nc1_z_metrics(z, labels)
        assert result["nc1_z"] == pytest.approx(0.0, abs=1e-10)
        assert result["r_z"] == pytest.approx(0.0, abs=1e-10)
        assert result["trace_w"] == pytest.approx(0.0, abs=1e-10)

    def test_nc1_z_trace_w_trace_b_ratio_matches_r_z(self):
        # trace_w/trace_b must equal r_z (SPEC3 section 18.1 names both
        # quantities separately; r_z is their ratio).
        z = np.array([[1.0, -1.0], [1.5, -0.5], [-1.0, 1.0], [-0.5, 1.5]])
        labels = np.array([0, 0, 1, 1])
        result = nc1_z_metrics(z, labels)
        assert result["trace_b"] > 0.0
        assert result["trace_w"] / result["trace_b"] == pytest.approx(result["r_z"], rel=1e-10)

    def test_margins_basic(self):
        z = np.array([[2.0, -1.0, 0.0], [-1.0, 3.0, 0.0]])
        labels = np.array([0, 1])
        result = margins(z, labels)
        # sample 0: 2.0 - max(-1.0, 0.0) = 2.0; sample 1: 3.0 - max(-1.0, 0.0) = 3.0
        assert result["mean_margin"] == pytest.approx(2.5, abs=1e-10)
        assert result["min_margin"] == pytest.approx(2.0, abs=1e-10)

    def test_d_box_zero_at_vertices(self):
        # C=3 box vertices: v_0=(1,-1,-1), v_1=(-1,1,-1), v_2=(-1,-1,1).
        zbar_c = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
        assert d_box(zbar_c) == pytest.approx(0.0, abs=1e-10)

    def test_equiangle_dev_z_hand_built_c3_etf_r3(self):
        # Standard-basis vertices in R^3 centered -> exact C=3 ETF (cos=-1/2).
        z = np.eye(3)
        labels = np.array([0, 1, 2])
        assert equiangle_dev_z(z, labels) == pytest.approx(0.0, abs=1e-10)

    def test_box_vertices_c3_cosine_equals_etf_target(self):
        """THEORY_NOTES T2' correction: at C=3, the centered box vertices
        (+1 in one coordinate, -1 elsewhere) have pairwise cosine exactly
        -1/(C-1) = -0.5 -- the SAME angular target as the simplex ETF.
        equiangle_dev_z (an angle-only witness) therefore cannot by itself
        distinguish the box-vertex model from the ETF model at C=3; the
        box model's distinguishing prediction is the RADIUS
        ||zbar_c - zbar_G|| (box: saturates near a fixed nonzero value;
        ETF + free beta: scale-free, no saturation), not the angle."""
        v_0 = np.array([1.0, -1.0, -1.0])
        v_1 = np.array([-1.0, 1.0, -1.0])
        v_2 = np.array([-1.0, -1.0, 1.0])
        global_mean = (v_0 + v_1 + v_2) / 3.0
        centered = [v_0 - global_mean, v_1 - global_mean, v_2 - global_mean]
        for i in range(3):
            for j in range(i + 1, 3):
                cos_ij = np.dot(centered[i], centered[j]) / (
                    np.linalg.norm(centered[i]) * np.linalg.norm(centered[j])
                )
                assert cos_ij == pytest.approx(-0.5, abs=1e-10)
        # equiangle_dev_z (treating each vertex as its own class's zbar) is
        # therefore also exactly 0 -- box vertices pass the angle-only test
        # trivially, confirming the angle test alone cannot discriminate.
        z = np.stack([v_0, v_1, v_2])
        labels = np.array([0, 1, 2])
        assert equiangle_dev_z(z, labels) == pytest.approx(0.0, abs=1e-10)

    def test_fiber_fraction_zero_for_pure_measured_direction(self):
        basis = [PAULI_Z / np.sqrt(2)]
        deviations = np.stack([PAULI_Z / np.sqrt(2)])
        result = fiber_decomposition(deviations, basis)
        assert result["fiber_fraction"] == pytest.approx(0.0, abs=1e-10)

    def test_fiber_fraction_one_for_orthogonal_direction(self):
        basis = [PAULI_Z / np.sqrt(2)]
        deviations = np.stack([PAULI_X])
        result = fiber_decomposition(deviations, basis)
        assert result["fiber_fraction"] == pytest.approx(1.0, abs=1e-10)

    def test_total_within_class_hs_variance_equals_v_meas_plus_v_fiber(self):
        rng = np.random.default_rng(0)
        basis = measured_subspace_basis(n_qubits=4, num_classes=3)
        deviations = []
        for _ in range(6):
            h = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
            h = (h + h.conj().T) / 2
            deviations.append(h)
        deviations = np.stack(deviations)

        direct = total_within_class_hs_variance(deviations)
        decomposed = fiber_decomposition(deviations, basis)
        assert direct == pytest.approx(decomposed["v_meas"] + decomposed["v_fiber"], rel=1e-10)

    def test_measured_subspace_basis_orthonormal_n4(self):
        basis = measured_subspace_basis(n_qubits=4, num_classes=3)
        for j in range(3):
            for k in range(3):
                expected = 1.0 if j == k else 0.0
                assert hs_inner(basis[j], basis[k]) == pytest.approx(expected, abs=1e-10)

    def test_loss_visible_split_zero_complement_when_all_variance_in_wrows(self):
        # k=2 measured span (B0=Z/sqrt2, B1=X/sqrt2, orthonormal single-qubit
        # Paulis). Head has 1 row picking out B0 only. Deviation is purely
        # along B0 -> the W-row image captures all of it, complement = 0.
        b0 = PAULI_Z / np.sqrt(2)
        b1 = PAULI_X / np.sqrt(2)
        basis = [b0, b1]
        head_weight = np.array([[1.0, 0.0]])  # (C=1, k=2), row = B0 direction
        deviations = np.stack([b0])
        result = loss_visible_subspace_decomposition(deviations, basis, head_weight)
        assert result["wrow_dim"] == 1
        assert result["complement_dim"] == 1
        assert result["v_complement"] == pytest.approx(0.0, abs=1e-10)
        assert result["v_wrow"] == pytest.approx(result["v_meas"], rel=1e-10)

    def test_loss_visible_split_proportional_to_dims_for_isotropic_input(self):
        # Same k=2 span, head still picks out B0 only (wrow_dim=1,
        # complement_dim=1). Isotropic input: one sample purely along B0
        # (fully in the W-row image), one purely along B1 (fully in the
        # complement, orthogonal to the head's single row) -> split is
        # exactly 50/50, proportional to wrow_dim/k = complement_dim/k = 1/2.
        b0 = PAULI_Z / np.sqrt(2)
        b1 = PAULI_X / np.sqrt(2)
        basis = [b0, b1]
        head_weight = np.array([[1.0, 0.0]])
        deviations = np.stack([b0, b1])
        result = loss_visible_subspace_decomposition(deviations, basis, head_weight)
        assert result["v_meas"] == pytest.approx(1.0, abs=1e-10)
        assert result["v_wrow"] == pytest.approx(0.5, abs=1e-10)
        assert result["v_complement"] == pytest.approx(0.5, abs=1e-10)

    def test_isotropic_fiber_null_matches_dimension_count(self):
        # n=4: total traceless dim = 4^4 - 1 = 255. k=9 (R1) -> 1 - 9/255.
        assert isotropic_fiber_null(4, 9) == pytest.approx(1 - 9 / 255, abs=1e-12)
        assert isotropic_fiber_null(4, 18) == pytest.approx(1 - 18 / 255, abs=1e-12)
        assert isotropic_fiber_null(4, 3) == pytest.approx(1 - 3 / 255, abs=1e-12)


class TestStage13ReadoutFamilyBases:
    """SPEC3_ADDENDUM.md section 19 (Stage 13): orthonormalization unit test
    per readout family arm, n_qubits=4, num_classes=3."""

    def test_r1_basis_orthonormal_9_operators(self):
        basis = measured_subspace_basis_r1(n_qubits=4, num_classes=3)
        assert len(basis) == 9
        for j in range(9):
            for k in range(9):
                expected = 1.0 if j == k else 0.0
                assert hs_inner(basis[j], basis[k]) == pytest.approx(expected, abs=1e-10)

    def test_r2_basis_orthonormal_18_operators(self):
        basis = measured_subspace_basis_r2(n_qubits=4)
        assert len(basis) == 18  # 3*4 weight-1 + 4*3/2 weight-2 ZZ
        for j in range(18):
            for k in range(18):
                expected = 1.0 if j == k else 0.0
                assert hs_inner(basis[j], basis[k]) == pytest.approx(expected, abs=1e-10)

    def test_r3_basis_orthonormal_for_fixed_random_phi(self):
        rng = np.random.default_rng(0)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        measurement_weights = rng.uniform(0, 2 * np.pi, size=weight_shape)
        basis = measured_subspace_basis_r3(n_qubits=4, num_classes=3, measurement_weights=measurement_weights)
        assert len(basis) == 3
        for j in range(3):
            for k in range(3):
                expected = 1.0 if j == k else 0.0
                assert hs_inner(basis[j], basis[k]) == pytest.approx(expected, abs=1e-10)
