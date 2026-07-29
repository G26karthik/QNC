"""Pure math metrics functions.

Implements SPEC.md §4 (quantum NC) and §5 (classical NC control). No I/O,
no globals. Every function ships with its SPEC §6 unit test, written and
passing before it is called from any experiment code (CLAUDE.md invariant 1).
"""

from typing import Callable

import numpy as np
import pennylane as qml
import torch


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """D_tr(rho, sigma) = (1/2) sum_k |lambda_k|, eigenvalues of (rho - sigma).

    Implements SPEC §4.2.
    """
    diff = rho - sigma
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def hs_inner(a: np.ndarray, b: np.ndarray) -> float:
    """<A, B> = Tr(A^dagger B). Real for Hermitian A, B.

    Implements SPEC §4.3. Asserts imaginary part < 1e-10 per SPEC.
    """
    val = np.trace(a.conj().T @ b)
    assert abs(val.imag) < 1e-10, f"hs_inner: imaginary part {val.imag} >= 1e-10"
    return float(val.real)


def hs_norm(a: np.ndarray) -> float:
    """||A||_HS = sqrt(Tr(A^dagger A)).

    Implements SPEC §4.2 (used to build ||rho_i - rho_bar_c||^2_HS).
    """
    return float(np.sqrt(hs_inner(a, a)))


def class_means(rhos: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rho_bar_c = (1/N_c) sum_{i in c} rho_i; rho_bar_G = (1/N) sum_i rho_i.

    Implements SPEC §4.1. Classes ordered by sorted unique label value.

    Returns (means: array shape (C, d, d), global_mean: array shape (d, d)).
    """
    classes = np.unique(labels)
    means = np.stack([rhos[labels == c].mean(axis=0) for c in classes])
    global_mean = rhos.mean(axis=0)
    return means, global_mean


def centered_means(means: np.ndarray, global_mean: np.ndarray) -> np.ndarray:
    """M_c = rho_bar_c - rho_bar_G.

    Implements SPEC §4.1.
    """
    return means - global_mean[None, :, :]


def qnc1_metrics(
    rhos: np.ndarray, labels: np.ndarray, means: np.ndarray, global_mean: np.ndarray
) -> dict:
    """Within-class concentration, two independent witnesses.

    Implements SPEC §4.2:
    QNC1_trace = (1/C) sum_c (1/N_c) sum_{i in c} D_tr(rho_i, rho_bar_c)
    QNC1_hs    = (1/C) sum_c (1/N_c) sum_{i in c} ||rho_i - rho_bar_c||^2_HS
    QNC1_rel   = QNC1_hs / mean_c ||M_c||^2_HS  (NaN if denominator < 1e-12)
    """
    classes = np.unique(labels)
    num_classes = len(classes)
    m = centered_means(means, global_mean)

    trace_total = 0.0
    hs_total = 0.0
    for idx, c in enumerate(classes):
        class_rhos = rhos[labels == c]
        rho_bar_c = means[idx]
        trace_total += np.mean(
            [trace_distance(rho_i, rho_bar_c) for rho_i in class_rhos]
        )
        hs_total += np.mean(
            [hs_norm(rho_i - rho_bar_c) ** 2 for rho_i in class_rhos]
        )

    qnc1_trace = trace_total / num_classes
    qnc1_hs = hs_total / num_classes

    mc_norms_sq = np.array([hs_norm(m[idx]) ** 2 for idx in range(num_classes)])
    denom = float(mc_norms_sq.mean())
    qnc1_rel = qnc1_hs / denom if denom >= 1e-12 else float("nan")

    return {"qnc1_trace": qnc1_trace, "qnc1_hs": qnc1_hs, "qnc1_rel": qnc1_rel}


def qnc1_fisher(qnc1_hs: float, class_mean_norms: np.ndarray) -> float:
    """QNC1_fisher = S_W / S_B, S_W = qnc1_hs, S_B = mean_c ||M_c||^2_HS.

    Implements SPEC2_ADDENDUM.md §11.1. Recomputable from JSONL alone: both
    inputs are already-logged Phase 1 fields (class_mean_norms are the HS
    norms of the centered class means). NaN if S_B < 1e-12 (SPEC2 §11.1
    mirrors the existing qnc1_rel guard).
    """
    s_b = float(np.mean(np.asarray(class_mean_norms, dtype=float) ** 2))
    return qnc1_hs / s_b if s_b >= 1e-12 else float("nan")


def qnc1_ratio(d_within: float, means: np.ndarray) -> float:
    """QNC1_ratio = D_within / D_between, D_between = mean_{c<c'} D_tr(rho_bar_c, rho_bar_c').

    Implements SPEC2_ADDENDUM.md §11.2. `d_within` is the already-computed
    qnc1_trace for the same epoch; `means` are the (uncentered) class-mean
    density matrices, shape (C, d, d). NaN if D_between < 1e-9 (SPEC2 guard).
    """
    num_classes = means.shape[0]
    pair_dists = [
        trace_distance(means[i], means[j])
        for i in range(num_classes)
        for j in range(i + 1, num_classes)
    ]
    d_between = float(np.mean(pair_dists))
    return d_within / d_between if d_between >= 1e-9 else float("nan")


def overlap_offdiag(means: np.ndarray) -> float:
    """Du-orthogonality witness: mean_{c<c'} Tr(rho_bar_c @ rho_bar_c').

    Implements SPEC2_ADDENDUM.md §11.3. `means` are UNCENTERED class-mean
    density matrices, shape (C, d, d). Tends to 0 for orthogonal supports
    (Du et al.'s prediction), to the mean purity for identical means.
    """
    num_classes = means.shape[0]
    overlaps = [
        hs_inner(means[i], means[j])
        for i in range(num_classes)
        for j in range(i + 1, num_classes)
    ]
    return float(np.mean(overlaps))


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2), a within-class concentration witness with full dynamic
    range (unlike raw trace distance, does not saturate near its ceiling
    for random high-dimensional states).

    Implements SPEC2_ADDENDUM.md §11.3.
    """
    assert np.allclose(rho, rho.conj().T, atol=1e-8), "purity: rho not Hermitian"
    val = np.trace(rho @ rho)
    assert abs(val.imag) < 1e-8, f"purity: imaginary part {val.imag} >= 1e-8"
    return float(val.real)


def qnc2_metrics(means: np.ndarray, global_mean: np.ndarray) -> dict:
    """Class-mean geometry, Simplex ETF test.

    Implements SPEC §4.3.
    """
    num_classes = means.shape[0]
    m = centered_means(means, global_mean)
    norms = np.array([hs_norm(m[c]) for c in range(num_classes)])

    gram = np.zeros((num_classes, num_classes))
    cos = np.full((num_classes, num_classes), np.nan)
    for i in range(num_classes):
        for j in range(num_classes):
            gram[i, j] = hs_inner(m[i], m[j])
            if i != j:
                cos[i, j] = gram[i, j] / (norms[i] * norms[j])

    mean_norm = float(norms.mean())
    equinorm_cv = float(norms.std() / mean_norm) if mean_norm >= 1e-12 else float("nan")

    etf_target = -1.0 / (num_classes - 1)
    off_diag = np.array(
        [
            cos[i, j]
            for i in range(num_classes)
            for j in range(i + 1, num_classes)
        ]
    )
    equiangle_dev = float(np.mean(np.abs(off_diag - etf_target)))
    equiangle_std = float(np.std(off_diag))

    return {
        "gram": gram,
        "cos": cos,
        "class_mean_norms": norms,
        "equinorm_cv": equinorm_cv,
        "equiangle_dev": equiangle_dev,
        "equiangle_std": equiangle_std,
        "etf_target": etf_target,
    }


def qfi_matrix(psi_fn: Callable[[], torch.Tensor], params: torch.Tensor) -> np.ndarray:
    """Quantum Fisher information matrix (real part of the quantum geometric
    tensor), averaged over a batch of states.

    Implements SPEC2_ADDENDUM.md §12.2:
    G_jk = Re[ <d_j psi|d_k psi> - <d_j psi|psi><psi|d_k psi> ]

    `psi_fn` is a zero-arg closure returning a differentiable complex torch
    tensor of shape (batch, dim) built from `params` (a leaf tensor with
    `requires_grad=True`, e.g. a model's variational weights); computed via
    autograd per SPEC2 §16. Returns a real (n_params, n_params) array.
    """
    psi = psi_fn()
    batch, dim = psi.shape
    n_params = params.numel()

    jac = np.zeros((batch, dim, n_params), dtype=complex)
    for b in range(batch):
        for k in range(dim):
            grad_real = torch.autograd.grad(psi[b, k].real, params, retain_graph=True)[0]
            grad_imag = torch.autograd.grad(psi[b, k].imag, params, retain_graph=True)[0]
            jac[b, k] = (grad_real + 1j * grad_imag).flatten().detach().numpy()

    psi_np = psi.detach().numpy()
    g_batch = np.zeros((batch, n_params, n_params))
    for b in range(batch):
        dpsi = jac[b]  # (dim, n_params)
        overlap = dpsi.conj().T @ dpsi
        a = dpsi.conj().T @ psi_np[b]
        proj = np.outer(a, a.conj())
        g_batch[b] = (overlap - proj).real

    return g_batch.mean(axis=0)


def qfi_rank(g: np.ndarray, rel_threshold: float = 1e-8) -> int:
    """rank(G) via singular values > rel_threshold * max singular value.

    Implements SPEC2_ADDENDUM.md §12.2.
    """
    singular_values = np.linalg.svd(g, compute_uv=False)
    threshold = rel_threshold * singular_values.max()
    return int(np.sum(singular_values > threshold))


def reduced_density(rho: np.ndarray, n_qubits: int, num_keep: int) -> np.ndarray:
    """rho_A = Tr_B(rho), B = last (n_qubits - num_keep) qubits.

    Implements SPEC §4.4 bipartition (A = first ceil(n/2) qubits, fixed for
    all runs; caller passes num_keep = ceil(n/2)).
    """
    dim_a = 2**num_keep
    dim_b = 2 ** (n_qubits - num_keep)
    tensor = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    return np.einsum("abcb->ac", tensor)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S(rho) = -sum_k lambda_k log2(lambda_k), 0*log(0) := 0.

    Implements SPEC §4.4. Eigenvalues clipped to [0, 1] after asserting they
    exceed -1e-10 (SPEC §9 numerical policy).
    """
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals > -1e-10), f"eigenvalue below -1e-10: {eigvals.min()}"
    eigvals = np.clip(eigvals, 0.0, 1.0)
    nonzero = eigvals[eigvals > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def classical_nc1(features: np.ndarray, labels: np.ndarray) -> float:
    """NC1 = (1/C) * tr(Sigma_W @ pinv(Sigma_B)).

    Implements SPEC §5. features: (N, p) penultimate-layer activations.
    """
    classes = np.unique(labels)
    num_classes = len(classes)
    num_samples, dim = features.shape

    sigma_w = np.zeros((dim, dim))
    class_means_arr = np.zeros((num_classes, dim))
    for idx, c in enumerate(classes):
        hc = features[labels == c]
        mu_c = hc.mean(axis=0)
        class_means_arr[idx] = mu_c
        diffs = hc - mu_c
        sigma_w += diffs.T @ diffs
    sigma_w /= num_samples

    mu_g = features.mean(axis=0)
    sigma_b = np.zeros((dim, dim))
    for idx in range(num_classes):
        d = (class_means_arr[idx] - mu_g).reshape(-1, 1)
        sigma_b += d @ d.T
    sigma_b /= num_classes

    nc1 = (1.0 / num_classes) * np.trace(sigma_w @ np.linalg.pinv(sigma_b))
    return float(nc1)


def within_class_scatter_trace(features: np.ndarray, labels: np.ndarray) -> float:
    """tr(Sigma_W), the ABSOLUTE (unnormalized) within-class scatter trace.

    Implements SPEC3_ADDENDUM.md section 24 -- the direct classical analogue
    of the VQC's total_var (section 18). Same Sigma_W construction as
    classical_nc1 (SPEC section 5), but reports the raw trace rather than
    normalizing by Sigma_B. features: (N, p) penultimate-layer activations.
    """
    classes = np.unique(labels)
    num_samples, dim = features.shape

    sigma_w = np.zeros((dim, dim))
    for c in classes:
        hc = features[labels == c]
        diffs = hc - hc.mean(axis=0)
        sigma_w += diffs.T @ diffs
    sigma_w /= num_samples

    return float(np.trace(sigma_w))


_PAULI_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_operator(n_qubits: int, qubit: int, pauli: str) -> np.ndarray:
    """Full 2^n x 2^n single-qubit Pauli operator I (x) ... (x) P_qubit (x)
    ... (x) I, pauli in {"X", "Y", "Z"}.

    Generalizes pauli_z_operator to all three Paulis. Implements
    SPEC3_ADDENDUM.md section 19 (R1/R2 measured-subspace features).
    """
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = _PAULI_MATRICES[pauli]
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full


def pauli_zz_operator(n_qubits: int, i: int, j: int) -> np.ndarray:
    """Full 2^n x 2^n weight-2 operator I (x) ... Z_i ... Z_j ... (x) I.

    Implements SPEC3_ADDENDUM.md section 19 (R2 weight-2 ZZ features).
    """
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[i] = _PAULI_MATRICES["Z"]
    ops[j] = _PAULI_MATRICES["Z"]
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full


def pauli_z_operator(n_qubits: int, qubit: int) -> np.ndarray:
    """Full 2^n x 2^n operator I (x) ... (x) Z_qubit (x) ... (x) I.

    Implements SPEC3_ADDENDUM.md section 18.4 ({A_1,...,A_C} = {Z_0,...,Z_{C-1}}).
    """
    return pauli_operator(n_qubits, qubit, "Z")


def z_features(rhos: np.ndarray, n_qubits: int, num_classes: int) -> np.ndarray:
    """z_i = ( <psi_i|Z_0|psi_i>, ..., <psi_i|Z_{C-1}|psi_i> ), shape (N, C).

    Implements SPEC3_ADDENDUM.md section 18 (z-space logit-feature vector),
    z_i[c] = Tr(rho_i Z_c).
    """
    z_ops = [pauli_z_operator(n_qubits, c) for c in range(num_classes)]
    n = rhos.shape[0]
    z = np.zeros((n, num_classes))
    for i in range(n):
        for c in range(num_classes):
            val = np.trace(rhos[i] @ z_ops[c])
            assert abs(val.imag) < 1e-8, f"z_features: imaginary part {val.imag} >= 1e-8"
            z[i, c] = val.real
    return z


def nc1_z_metrics(z: np.ndarray, labels: np.ndarray) -> dict:
    """NC1_z and r_z (SPEC3_ADDENDUM.md section 18.1).

    NC1_z = (1/C) tr(Sigma_W^z pinv(Sigma_B^z))
    r_z   = tr(Sigma_W^z) / tr(Sigma_B^z)  (scale-free scalar ratio)

    Also returns trace_w = tr(Sigma_W^z) and trace_b = tr(Sigma_B^z)
    separately (within-class shrink vs between-class growth), both named
    explicitly in section 18.1.
    """
    classes = np.unique(labels)
    num_classes = len(classes)
    num_samples, dim = z.shape

    sigma_w = np.zeros((dim, dim))
    class_means_arr = np.zeros((num_classes, dim))
    for idx, c in enumerate(classes):
        zc = z[labels == c]
        mu_c = zc.mean(axis=0)
        class_means_arr[idx] = mu_c
        diffs = zc - mu_c
        sigma_w += diffs.T @ diffs
    sigma_w /= num_samples

    mu_g = z.mean(axis=0)
    sigma_b = np.zeros((dim, dim))
    for idx in range(num_classes):
        d = (class_means_arr[idx] - mu_g).reshape(-1, 1)
        sigma_b += d @ d.T
    sigma_b /= num_classes

    nc1_z = float((1.0 / num_classes) * np.trace(sigma_w @ np.linalg.pinv(sigma_b)))
    trace_w = float(np.trace(sigma_w))
    trace_b = float(np.trace(sigma_b))
    r_z = trace_w / trace_b if trace_b >= 1e-12 else float("nan")

    return {
        "nc1_z": nc1_z,
        "r_z": r_z,
        "trace_w": trace_w,
        "trace_b": trace_b,
        "zbar_c": class_means_arr,
        "zbar_g": mu_g,
    }


def margins(z: np.ndarray, labels: np.ndarray) -> dict:
    """Per-sample logit margin m_i = z_i[y_i] - max_{c != y_i} z_i[c].

    Implements SPEC3_ADDENDUM.md section 18.2. Returns mean_margin, min_margin.
    """
    n = z.shape[0]
    m = np.zeros(n)
    for i in range(n):
        y = int(labels[i])
        others = np.delete(z[i], y)
        m[i] = z[i, y] - others.max()
    return {"mean_margin": float(m.mean()), "min_margin": float(m.min())}


def d_box(zbar_c: np.ndarray) -> float:
    """d_box = (1/C) sum_c ||zbar_c - v_c||_2, v_c the box vertex

    (+1 in coordinate c, -1 elsewhere). Implements SPEC3_ADDENDUM.md section
    18.3. `zbar_c` shape (C, C).
    """
    num_classes = zbar_c.shape[0]
    total = 0.0
    for c in range(num_classes):
        v_c = -np.ones(num_classes)
        v_c[c] = 1.0
        total += float(np.linalg.norm(zbar_c[c] - v_c))
    return total / num_classes


def equiangle_dev_z(z: np.ndarray, labels: np.ndarray) -> float:
    """d_etf = equiangle_dev_z (SPEC3_ADDENDUM.md section 18.3): pairwise
    cosine deviation from the ETF target -1/(C-1), Euclidean inner product in
    R^C. Algebraically identical to classical_nc2's equiangle_dev (SPEC
    section 4.3 generalized to Euclidean means); a dedicated wrapper matching
    the SPEC3 field name. Applies at C=3 only per SPEC3 section 18.3.
    """
    return classical_nc2(z, labels)["equiangle_dev"]


def measured_subspace_basis(n_qubits: int, num_classes: int) -> list[np.ndarray]:
    """Orthonormalized Pauli-Z readout basis B_j = Z_j / 2^(n/2), j=0..C-1.

    Implements SPEC3_ADDENDUM.md section 18.4. Orthonormality (<B_j,B_k>_HS =
    delta_jk) follows from <Z_j,Z_k>_HS = delta_jk * 2^n.
    """
    scale = 2 ** (n_qubits / 2)
    return [pauli_z_operator(n_qubits, c) / scale for c in range(num_classes)]


def measured_subspace_basis_r1(n_qubits: int, num_classes: int) -> list[np.ndarray]:
    """R1 measured-subspace basis: all 3 Paulis on each of the first
    `num_classes` qubits (3*C operators), orthonormalized by 2^(n/2).
    Distinct Pauli strings are automatically HS-orthogonal (Pauli algebra),
    verified by unit test. Feature order matches VQC._expval_circuit's R1
    branch (qubit-major, X/Y/Z per qubit). Implements SPEC3_ADDENDUM.md
    section 19.
    """
    scale = 2 ** (n_qubits / 2)
    basis = []
    for c in range(num_classes):
        for p in ("X", "Y", "Z"):
            basis.append(pauli_operator(n_qubits, c, p) / scale)
    return basis


def measured_subspace_basis_r2(n_qubits: int) -> list[np.ndarray]:
    """R2 measured-subspace basis: all weight-1 Paulis on all qubits (3n
    operators) plus all weight-2 ZZ pairs (n(n-1)/2 operators), orthonormalized
    by 2^(n/2). Order matches VQC._expval_circuit's R2 branch (weight-1 all
    qubits, then i<j ZZ pairs). Implements SPEC3_ADDENDUM.md section 19.
    """
    scale = 2 ** (n_qubits / 2)
    basis = []
    for q in range(n_qubits):
        for p in ("X", "Y", "Z"):
            basis.append(pauli_operator(n_qubits, q, p) / scale)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            basis.append(pauli_zz_operator(n_qubits, i, j) / scale)
    return basis


def measured_subspace_basis_r3(
    n_qubits: int, num_classes: int, measurement_weights: np.ndarray
) -> list[np.ndarray]:
    """R3 measured-subspace basis: fixed Z_c conjugated by the trained
    measurement unitary V(phi) = StronglyEntanglingLayers(measurement_weights),
    B_c = V(phi)^dagger Z_c V(phi) / 2^(n/2). HS-orthogonality is preserved
    under unitary conjugation (Tr(A^dagger B) is basis-independent), so this
    stays orthonormal for any phi -- verified by unit test at a fixed random
    phi. Implements SPEC3_ADDENDUM.md section 19.
    """
    wires = list(range(n_qubits))
    v = qml.matrix(qml.StronglyEntanglingLayers, wire_order=wires)(measurement_weights, wires=wires)
    scale = 2 ** (n_qubits / 2)
    basis = []
    for c in range(num_classes):
        z_c = pauli_z_operator(n_qubits, c)
        basis.append((v.conj().T @ z_c @ v) / scale)
    return basis


def fiber_decomposition(deviations: np.ndarray, basis: list[np.ndarray]) -> dict:
    """V_meas, V_fiber, fiber_fraction (SPEC3_ADDENDUM.md section 18.4).

    `deviations`: array (N, d, d) of D_i = rho_i - rho_bar_c (Hermitian).
    `basis`: orthonormal Hermitian operators spanning the measured subspace.
    """
    n = deviations.shape[0]
    v_meas_total = 0.0
    v_fiber_total = 0.0
    for i in range(n):
        d_i = deviations[i]
        d_meas = np.zeros_like(d_i)
        for b_j in basis:
            coeff = hs_inner(b_j, d_i)
            d_meas = d_meas + coeff * b_j
        d_fiber = d_i - d_meas
        v_meas_total += hs_norm(d_meas) ** 2
        v_fiber_total += hs_norm(d_fiber) ** 2
    v_meas = v_meas_total / n
    v_fiber = v_fiber_total / n
    denom = v_meas + v_fiber
    fiber_fraction = v_fiber / denom if denom >= 1e-12 else float("nan")
    return {"v_meas": v_meas, "v_fiber": v_fiber, "fiber_fraction": fiber_fraction}


def total_within_class_hs_variance(deviations: np.ndarray) -> float:
    """Direct, basis-independent total within-class Hilbert-Schmidt variance:
    (1/N) sum_i ||D_i||^2_HS, D_i = rho_i - rho_bar_c. By Parseval's identity
    this equals fiber_decomposition(deviations, basis)['v_meas'] +
    ['v_fiber'] for ANY orthonormal basis of the measured subspace, since the
    measured span and its fiber complement together span every traceless
    Hermitian direction. Implements PREDICTIONS.md's Stage 14B (HR1/HR2)
    rotation-invariance check.
    """
    n = deviations.shape[0]
    total = 0.0
    for i in range(n):
        total += hs_norm(deviations[i]) ** 2
    return total / n


def isotropic_fiber_null(n_qubits: int, measured_dim: int) -> float:
    """Dimension-counting null for fiber_fraction under an isotropic
    (untrained-like) within-class covariance: the measured subspace is
    `measured_dim` of the 4^n_qubits - 1 traceless Hermitian directions, so an
    isotropic deviation puts exactly that fraction of its variance in the
    fiber. Implements PREDICTIONS.md's Stage 13B section (1 - k/(4^n - 1))."""
    total_dim = 4**n_qubits - 1
    return 1.0 - measured_dim / total_dim


def loss_visible_subspace_decomposition(
    deviations: np.ndarray, basis: list[np.ndarray], head_weight: np.ndarray
) -> dict:
    """Splits within-measured-span variance (this module's `fiber_decomposition`
    V_meas) into the C-dim loss-visible image of the trained head's row space
    and its (k-C)-dim in-span complement.

    Implements PLAYBOOK3.md Stage 13B H1 / PREDICTIONS.md's Stage 13B section:
    for R1/R2, CE sees the state only through logits = W f where f is the
    k-dim measured-Pauli feature vector. This maps W's C rows into operator
    space via `basis` (each row w_c -> O_c = sum_j w_c[j] * basis[j]),
    Gram-Schmidt-orthonormalizes {O_c} within the measured span (rows may be
    linearly dependent for a trained head; near-zero directions are dropped),
    and projects each measured-subspace deviation onto that image vs its
    in-span complement.

    `deviations`: (N, d, d) array of D_i = rho_i - rho_bar_c (Hermitian).
    `basis`: orthonormal Hermitian operators spanning the measured subspace
    (length k), HS-inner-product order matching `head_weight`'s columns.
    `head_weight`: (C, k) trained linear head weight matrix.
    """
    k = len(basis)
    num_rows = head_weight.shape[0]
    row_ops = []
    for c in range(num_rows):
        op = np.zeros_like(basis[0])
        for j in range(k):
            op = op + head_weight[c, j] * basis[j]
        row_ops.append(op)

    ortho: list[np.ndarray] = []
    for op in row_ops:
        v = op.copy()
        for b in ortho:
            v = v - hs_inner(b, v) * b
        norm = hs_norm(v)
        if norm > 1e-10:
            ortho.append(v / norm)

    n = deviations.shape[0]
    v_meas_total = 0.0
    v_wrow_total = 0.0
    for i in range(n):
        d_i = deviations[i]
        d_meas = np.zeros_like(d_i)
        for b_j in basis:
            d_meas = d_meas + hs_inner(b_j, d_i) * b_j
        v_meas_total += hs_norm(d_meas) ** 2
        d_wrow = np.zeros_like(d_i)
        for b in ortho:
            d_wrow = d_wrow + hs_inner(b, d_i) * b
        v_wrow_total += hs_norm(d_wrow) ** 2

    v_meas = v_meas_total / n
    v_wrow = v_wrow_total / n
    v_complement = v_meas - v_wrow
    wrow_dim = len(ortho)
    complement_dim = k - wrow_dim
    return {
        "v_meas": v_meas,
        "v_wrow": v_wrow,
        "v_complement": v_complement,
        "wrow_dim": wrow_dim,
        "complement_dim": complement_dim,
    }


def classical_nc2(features: np.ndarray, labels: np.ndarray) -> dict:
    """NC2 equinorm/equiangularity with Euclidean inner product on mu_c - mu_G.

    Implements SPEC §5 (same formulas as §4.3, Euclidean instead of HS).
    """
    classes = np.unique(labels)
    num_classes = len(classes)
    class_means_arr = np.stack(
        [features[labels == c].mean(axis=0) for c in classes]
    )
    mu_g = features.mean(axis=0)
    m = class_means_arr - mu_g[None, :]

    norms = np.linalg.norm(m, axis=1)
    gram = m @ m.T
    cos = np.full((num_classes, num_classes), np.nan)
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                cos[i, j] = gram[i, j] / (norms[i] * norms[j])

    mean_norm = float(norms.mean())
    equinorm_cv = float(norms.std() / mean_norm) if mean_norm >= 1e-12 else float("nan")

    etf_target = -1.0 / (num_classes - 1)
    off_diag = np.array(
        [
            cos[i, j]
            for i in range(num_classes)
            for j in range(i + 1, num_classes)
        ]
    )
    equiangle_dev = float(np.mean(np.abs(off_diag - etf_target)))
    equiangle_std = float(np.std(off_diag))

    return {
        "gram": gram,
        "cos": cos,
        "class_mean_norms": norms,
        "equinorm_cv": equinorm_cv,
        "equiangle_dev": equiangle_dev,
        "equiangle_std": equiangle_std,
        "etf_target": etf_target,
    }
