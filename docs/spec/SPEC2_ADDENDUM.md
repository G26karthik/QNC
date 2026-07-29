# SPEC2_ADDENDUM.md -- Phase 2 Mathematical Specification
# Read together with SPEC.md. SPEC.md sections 1-9 remain in force.
# This addendum defines everything new for the overparameterization-transition
# study. Same rule applies: code must implement these formulas exactly.

---

## 10. Motivation for Phase 2 (context the agent must internalize)

Phase 1 found: (a) C=3 never reached TPT across 17 tuning attempts at 6 qubits
with 72 parameters; (b) terminal qnc1_trace was statistically indistinguishable
from an untrained null baseline.

Both findings now have candidate explanations from the literature:

(a) Larocca, Ju, Garcia-Martin, Coles, Cerezo, "Theory of overparametrization
in quantum neural networks," Nature Computational Science 3, 542-551 (2023):
underparametrized QNNs have spurious local minima that disappear when the
parameter count M exceeds a critical M_c, upper-bounded by the dimension of
the dynamical Lie algebra (DLA) of the ansatz generators. For a generic
entangling ansatz on n qubits the DLA is close to full su(2^n), dimension
4^n - 1. Phase 1 ran at M = 72 vs 4^6 - 1 = 4095: about 1.8% of the bound.
The C=3 failure is predicted by this theory.

(b) Concentration of measure: random pure states in dimension 2^n are nearly
maximally trace-distant from one another, so UNNORMALIZED within-class trace
distance has almost no dynamic range near its ceiling. The Phase 1 null result
may be an artifact of the unnormalized metric. Section 11 defines normalized
witnesses; Stage 7 recomputes the headline from existing logs before any new
runs.

Phase 2's central question: **Is quantum neural collapse an order parameter of
the overparameterization phase transition?** Prediction to test: TPT
reachability and collapse metrics improve sharply at M ~ M_c.

New mandatory citation (add to SPEC section 0.4 list):
- Larocca et al. (2023), Nature Computational Science 3, 542-551
- Bowles, Ahmed, Schuld (2024), arXiv:2403.07059 (benchmark datasets + the
  entanglement-is-dispensable finding our ablation is in tension with)
- Han, Papyan, Donoho (2022), ICLR (NC under MSE loss, classical dynamics
  precedent)

---

## 11. Normalized collapse witnesses (Stage 7 -- recomputable from Phase 1 data)

### 11.1 Fisher-style HS ratio (recomputable from JSONL alone)

```
S_W = (1/C) sum_c (1/N_c) sum_{i in c} ||rho_i - rho_bar_c||^2_HS   (= qnc1_hs, already logged)
S_B = (1/C) sum_c ||M_c||^2_HS                                       (M_c = rho_bar_c - rho_bar_G)
QNC1_fisher = S_W / S_B
```

`class_mean_norms` is already in the Phase 1 JSONL, so S_B and QNC1_fisher are
recomputable per logged epoch WITHOUT touching checkpoints. This is the direct
analog of classical NC1 (within/between ratio) and is the new primary witness.

### 11.2 Trace-distance ratio (needs checkpoints)

```
D_within  = (1/C) sum_c (1/N_c) sum_{i in c} D_tr(rho_i, rho_bar_c)   (= qnc1_trace)
D_between = mean_{c<c'} D_tr(rho_bar_c, rho_bar_c')
QNC1_ratio = D_within / D_between
```

Recompute per checkpointed epoch by reloading parameters and regenerating
states. Guard: if D_between < 1e-9, report NaN.

### 11.3 Du-orthogonality witness (direct test of Du et al.'s prediction)

Du et al. predict UNCENTERED class-mean states become mutually orthogonal at
the optimum. Test directly:

```
overlap_offdiag = mean_{c<c'} Tr(rho_bar_c rho_bar_c')
```

For orthogonal supports this tends to 0; for identical means it equals
Tr(rho_bar^2) (the purity). Also log per-class purity Tr(rho_bar_c^2) --
within-class collapse to a pure state drives purity toward 1, an independent
concentration witness with full dynamic range (random-mean purity at n qubits
is far below 1, so this metric does NOT saturate like raw trace distance).

### 11.4 Required unit tests (add to tests/test_metrics.py)

| Test | Expected |
|---|---|
| QNC1_fisher with all rho_i == rho_bar_c per class, distinct means | 0.0 |
| QNC1_ratio same setup | 0.0 |
| overlap_offdiag for rho_bar_0 = <0><0|, rho_bar_1 = |1><1| | 0.0 |
| overlap_offdiag for identical pure means | 1.0 |
| purity of a pure state | 1.0 |
| purity of maximally mixed on 2 qubits | 0.25 |

## 12. Parameter counting and the M_c sweep (Stage 9 headline)

### 12.1 Setup

Primary architecture: n = 4 qubits (Hilbert dim 16, full su(16) dim = 255).
SEL on 4 qubits has 3 rotation params x 4 qubits = 12 params/layer, so
M = 12L. DLA upper bound M_c <= 255 => L_c <= ~21.3.

Sweep: L in {2, 4, 8, 12, 16, 21, 26, 32}  (M in {24, 48, 96, 144, 192, 252, 312, 384})
spanning ~0.1 M_c to ~1.5 M_c. 5 seeds per L. Log M and M/M_c_bound in every
JSONL line (new fields: "n_params": int, "m_over_mc": float with
M_c_bound = 4^n - 1).

### 12.2 Empirical transition locator: QFI rank

At initialization and at final epoch, compute the quantum Fisher information
matrix rank (Larocca et al.'s empirical criterion: capacity saturates when
QFI rank stops growing with M).

For pure states the QFI is, up to a constant factor, the real part of the
quantum geometric tensor:

```
G_jk = Re[ <d_j psi | d_k psi> - <d_j psi | psi><psi | d_k psi> ]
```

with |d_j psi> = partial derivative of the output state wrt parameter j,
computed by parameter-shift or autograd on the statevector. Average G over
16 training inputs, then:

```
qfi_rank = rank(G_avg)   using threshold: singular values > 1e-8 * max_sv
```

Log qfi_rank at epoch 0 and final epoch for every run in the L sweep.
Prediction: qfi_rank grows with M then saturates near 255; the saturation
point is the empirical M_c.

Unit test: for a 1-qubit circuit RY(theta)|0>, QFI is 1x1 with G_11 = 1/4
(state derivative norm formula); assert to 1e-8. For two redundant parameters
RY(a)RY(b)|0>, rank(G) = 1.

### 12.3 Outcome variables vs M

For each L: fraction of seeds reaching tpt_reached (primary), E_0 distribution,
final QNC1_fisher, QNC1_ratio, overlap_offdiag, mean class purity,
equiangle_dev (meaningful now: C=3), entropy_mean, qfi_rank.

Headline figure: all of these vs M/M_c_bound on one aligned x-axis, with the
qfi_rank saturation point marked. This is the "collapse as order parameter"
plot.

## 13. Datasets (Stage 8)

Phase 2 drops PCA-MNIST as primary (criticized by Bowles et al. 2024 as a
weak QML benchmark). New dataset module requirements, all generated locally
and deterministically per seed (no network dependency):

1. **blobs3** (primary for the M_c sweep): 3 isotropic Gaussian blobs in R^4,
   config-controlled centers and cluster std (default: centers at distance 3.0,
   std 0.8, 60 samples/class). Rationale: interpolation difficulty is
   controllable, guaranteeing the sweep measures landscape effects, not data
   pathology. Config: `dataset: blobs3`, fields `blob_dist`, `blob_std`,
   `samples_per_class`.
2. **bars_stripes_4x4**: 4x4 binary bars-and-stripes with Gaussian pixel noise
   (std 0.5, following Bowles et al.), C=2. 16 pixels = exactly 2^4 amplitudes:
   supports both angle encoding (PCA-4) and amplitude encoding on 4 qubits
   (config `encoding: angle|amplitude`; amplitude = L2-normalized pixel vector
   via qml.AmplitudeEmbedding with normalize=True).
3. **linearly_separable_4d**: Bowles et al.'s "fruit fly" -- uniform hypercube
   samples split by the hyperplane orthogonal to (1,1,1,1), margin 0.1, C=2.
4. **mnist_pca4** retained ONLY as a secondary external-validity arm (digits
   0/1/2, PCA-4).

## 14. MSE loss arm (Stage 10)

Du et al.'s theorem covers regularized squared error, not cross-entropy.
Add config `loss: ce | mse`:

```
mse:  L = (1/N) sum_i || beta * z(x_i) - y_i ||^2 + lambda ||theta||^2
```

with y_i one-hot in {0,1}^C, z the vector of Pauli-Z expectations, beta
learnable as in SPEC section 2. TPT/E_0 defined identically (train accuracy
via argmax). Every Stage 10 comparison runs matched CE and MSE arms with
identical seeds/configs otherwise.

## 15. New JSONL fields (extend SPEC section 7 schema)

```
"qnc1_fisher": float, "qnc1_ratio": float|null,
"overlap_offdiag": float, "purity_class": [float], "purity_mean": float,
"n_params": int, "m_over_mc": float, "qfi_rank": int|null,
"loss_type": "ce|mse", "dataset": str, "encoding": "angle|amplitude"
```

qnc1_ratio is null on epochs without checkpoint recompute; qfi_rank is null
except epoch 0 and final. All Phase 1 figures gain QNC1_fisher panels; the
Stage 9 sweep gets the order-parameter figure of section 12.3.

## 16. Numerical policy additions

- QFI via autograd on the statevector (torch) preferred over parameter-shift
  for speed at n=4; validate the two agree to 1e-6 on one config (unit test).
- The L=32 runs have 384 params on dim-16 states: still trivial on CPU. If a
  full sweep exceeds ~12h wall time, halve samples_per_class before reducing
  seeds; never reduce seeds below 5 for the headline sweep.
- Purity computed as Tr(rho_bar_c @ rho_bar_c).real with Hermiticity assert.
