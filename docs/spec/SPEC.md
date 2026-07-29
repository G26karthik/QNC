# SPEC.md -- Quantum Neural Collapse: Mathematical Specification

This file is the **single source of truth** for all mathematical definitions in this
project. Code must implement these formulas exactly. If code and SPEC disagree, the
code is wrong. The agent must never improvise, "simplify", or substitute a formula.

---

## 0. Prior work and positioning

This section is mandatory reading. The agent must never describe QNC as
"entirely open" or "unexplored." The core theoretical result already exists.

### 0.1 Du et al. (2023), Physical Review Letters 131, 140601

"Problem-Dependent Power of Quantum Neural Networks on Multiclass
Classification" (Yuxuan Du, Yibo Yang, Dacheng Tao, Min-Hsiu Hsieh).

**What they proved:** When the empirical risk of a quantum classifier
approaches zero, (a) feature states concentrate around per-class means
with vanishing variability, and (b) the class-mean measurement operators
become mutually orthogonal. This orthogonal structure is the global
optimizer of the empirical risk. Since an orthogonal frame can be
trivially centered into a simplex ETF, classical and quantum NC are
structurally equivalent at the optimum.

**Their setting:** regularized squared-mean-error loss; unconstrained
features model (analogous to the classical UFM). Theoretical proof, not
empirical dynamics.

### 0.2 San Sebastian, Canizo & Orus (Feb 2026), arXiv:2602.08485

"Empirical Study of Observable Sets in Multiclass Quantum Classification."

**What they did:** Extended the NC analysis empirically to quantum
classifiers with cross-entropy and other losses, studying how different
observable set choices (Pauli strings vs computational-basis projectors)
affect the tendency toward NC. They confirm NC tendency across losses
but focus on observable-set comparison, not density-matrix geometry,
training dynamics, or entanglement.

### 0.3 What remains open (our contributions)

Du et al. characterize the **static optimum**; they do not study
**training dynamics** (how and when the VQC traverses from random
initialization toward that optimum). No prior work has:

1. **Tracked the epoch-by-epoch trajectory** of quantum collapse metrics
   (when does QNC1 begin to decay relative to TPT onset? In what order
   do the NC properties emerge?)
2. **Used trace distance** (a properly quantum-information-theoretic
   metric with operational meaning: optimal state-discrimination
   probability) as the primary within-class concentration measure.
   Du et al. and San Sebastian work in measurement-operator space,
   not density-matrix space.
3. **Studied entanglement's causal role** via controlled ablation
   (entangling vs non-entangling ansatz). If collapse occurs without
   entanglement, it is representation-geometric; if only with
   entanglement, something quantum-information-theoretically
   fundamental is at play.
4. **Measured entanglement entropy dynamics during collapse.**
   Does entanglement grow, shrink, or reorganize during TPT?
5. **Studied the interaction of learnable logit temperature (beta)
   with collapse geometry** under cross-entropy loss.

### 0.4 Mandatory citations

Every writeup, abstract, or results document produced by this project
must cite:

- Papyan, Han, Donoho (2020), PNAS 117(40):24652-24663
- Du, Yang, Tao, Hsieh (2023), PRL 131, 140601
- San Sebastian, Canizo, Orus (2026), arXiv:2602.08485

---

## 1. Notation

- `n` -- number of qubits. Hilbert space dimension `d = 2^n`.
- `C` -- number of classes.
- `N_c` -- number of samples in class `c`; `N = sum_c N_c`.
- `|psi_i>` -- output statevector of the VQC for sample `i` (before measurement).
- `rho_i = |psi_i><psi_i|` -- density matrix of sample `i` (pure, rank 1).
- All logs use natural units except entropy, which uses **log base 2** (bits).

## 2. Model (Stage 2)

- **Encoding:** angle encoding. Input `x in R^n` (PCA-reduced, standardized, then
  scaled to `[-pi, pi]` per feature using train-set min/max). Applied as
  `RY(x_j)` on qubit `j`. Optional data re-uploading: re-apply encoding before
  each variational block (config flag `reupload: true/false`).
- **Ansatz:** `qml.StronglyEntanglingLayers` with `L` layers (config).
  Ablation ansatz: same rotations with **all entangling gates removed**
  (config flag `entangling: false`).
- **Readout:** `z_c = <psi| Z_c |psi>` for `c = 0..C-1` (Pauli-Z on qubit `c`).
  Requires `C <= n`.
- **Logits:** `logit_c = beta * z_c` where `beta` is a **learnable inverse-temperature
  scalar**, initialized at `beta_0` (config, default 5.0). This is essential:
  raw expectations live in `[-1, 1]` and cannot drive cross-entropy toward
  collapse without scaling. Log `beta` every interval.
  (Note: this mechanism was independently identified as "Quantum Measurement
  Temperature" by arXiv:2606.22551, confirming the design choice.)
- **Loss:** softmax cross-entropy. Optional L2 penalty `lambda * ||theta||^2` on
  variational angles (config `weight_decay`, default 0.0; sweep later --
  classical NC literature shows weight decay strengthens collapse, but for
  angles the analog is nontrivial, so treat as an experimental knob).
- **Simulator:** `default.qubit`, exact statevector (`shots: null`).
  Finite shots only in Stage 5 robustness runs.

## 3. Terminal Phase of Training (TPT)

- `E_0` = first epoch with **train error = 0** (100% train accuracy).
- Training must continue to `max(total_epochs, 5 * E_0)`.
- If train error never reaches 0 within the epoch budget, the run is marked
  `tpt_reached: false` and is **invalid for NC conclusions** (still logged).

## 4. Quantum NC metrics (Stage 3)

### 4.1 Class means

```
rho_bar_c = (1/N_c) sum_{i in c} rho_i          (mixed in general)
rho_bar_G = (1/N)  sum_i rho_i                   (global mean)
M_c  = rho_bar_c - rho_bar_G                     (centered class mean)
```

Runtime assertions (tolerance 1e-8): every `rho` Hermitian, `Tr rho = 1`,
eigenvalues >= -1e-10; every `M_c` Hermitian and **traceless**.

### 4.2 QNC1 -- within-class concentration (two independent witnesses)

Trace distance: `D_tr(rho, sigma) = 0.5 * sum_k |lambda_k|`, where `lambda_k` are eigenvalues of
the Hermitian matrix `rho - sigma`.

```
QNC1_trace = (1/C) sum_c (1/N_c) sum_{i in c} D_tr(rho_i, rho_bar_c)
QNC1_hs    = (1/C) sum_c (1/N_c) sum_{i in c} ||rho_i - rho_bar_c||^2_HS
```

with `||A||^2_HS = Tr(A^dag A)`. Both must be reported; collapse is claimed only
if **both** decay.

Normalized variant (report alongside, for cross-config comparison):
`QNC1_rel = QNC1_hs / mean_c ||M_c||^2_HS`. Guard: if denominator < 1e-12,
report NaN, never divide.

### 4.3 QNC2 -- class-mean geometry (Simplex ETF test)

Hilbert-Schmidt inner product: `<A, B> = Tr(A^dag B)` (real for Hermitian A, B --
assert imaginary part < 1e-10 then take real part).

```
G_{cc'}      = <M_c, M_c'>                       (Gram matrix, CxC)
cos_theta_{cc'}   = G_{cc'} / (||M_c||_HS * ||M_c'||_HS),  c != c'
equinorm_cv  = std_c(||M_c||_HS) / mean_c(||M_c||_HS)
etf_target   = -1/(C-1)
equiangle_dev = mean_{c<c'} | cos_theta_{cc'} - etf_target |
equiangle_std = std_{c<c'} ( cos_theta_{cc'} )
```

Note for writeup: centered means live in the real vector space of traceless
Hermitian operators, dimension `d^2 - 1 = 4^n - 1`, so the ETF question is
well-posed. A Simplex ETF of C points requires `C - 1 <= 4^n - 1` -- always
satisfied here.

### 4.4 Entanglement entropy

Bipartition `A` = first `ceil(n/2)` qubits, `B` = rest (fixed for all runs).

```
rho_A(i) = Tr_B( rho_i )
S(rho_A) = - sum_k lambda_k log_2 lambda_k     (eigenvalues lambda_k of rho_A; 0*log 0 := 0)
entropy_mean      = (1/N) sum_i S(rho_A(i))
entropy_class[c]  = (1/N_c) sum_{i in c} S(rho_A(i))
entropy_classmean[c] = S( Tr_B( rho_bar_c ) )
```

## 5. Classical NC metrics (Stage 1 control)

Features `h_i in R^p` = penultimate-layer activations. Class means `mu_c`,
global mean `mu_G`.

```
Sigma_W = (1/N) sum_c sum_{i in c} (h_i - mu_c)(h_i - mu_c)^T
Sigma_B = (1/C) sum_c (mu_c - mu_G)(mu_c - mu_G)^T
NC1 = (1/C) * tr( Sigma_W * pinv(Sigma_B) )
```

NC2 equinorm/equiangularity: same formulas as section 4.3 with Euclidean inner
product on `mu_c - mu_G`. Model: MLP 784->512->512->p(=64)->C, ReLU, SGD momentum
0.9, weight decay 5e-4, cross-entropy, subset of MNIST (C=3 or 4 classes,
500 samples/class), trained far past zero train error.

## 6. Required unit tests (exact expected values)

Every function in `metrics.py` ships with these tests BEFORE any experiment:

| Test | Expected |
|---|---|
| `D_tr(|0><0|, |0><0|)` | 0.0 |
| `D_tr(|0><0|, |1><1|)` | 1.0 |
| `D_tr(|0><0|, |+><+|)` | `1/sqrt(2) = 0.70710678` |
| Pure-state identity: `D_tr = sqrt(1-|<psi|phi>|^2)` on 20 random pure-state pairs | agree to 1e-8 |
| `S(Tr_B |00><00|)` | 0.0 |
| `S(Tr_B Bell)` where Bell = `(|00>+|11>)/sqrt(2)` | 1.0 (bit) |
| `S(Tr_B GHZ_4)` half-partition | 1.0 (bit) |
| `<X, Z>_HS` (Pauli matrices) | 0.0 |
| `||X||_HS` | `sqrt(2)` |
| Hand-built C=3 Simplex ETF in R^2 fed through QNC2 code path | `equiangle_dev = 0`, `equinorm_cv = 0`, cosines `= -0.5` |
| `M_c` from random states | traceless, Hermitian to 1e-10 |
| Classical NC1 on synthetic data with known Sigma_W, Sigma_B | analytic value to 1e-6 |

## 7. Logging schema (JSONL, one line per logged epoch)

```json
{"run_id": str, "epoch": int, "split": "train|test",
 "loss": float, "acc": float, "beta": float,
 "qnc1_trace": float, "qnc1_hs": float, "qnc1_rel": float,
 "equinorm_cv": float, "equiangle_dev": float, "equiangle_std": float,
 "gram_offdiag_cos": [float], "class_mean_norms": [float],
 "entropy_mean": float, "entropy_class": [float], "entropy_classmean": [float],
 "tpt_reached": bool, "e0_epoch": int|null,
 "seed": int, "git_sha": str, "wall_time_s": float}
```

Metrics computed on the **train split** define collapse (per Papyan); also
compute on test split for the generalization story. Log every
`log_interval` epochs (config, default: every epoch for first 50, then every 5).
Checkpoint parameters at every logged epoch to `results/<run_id>/checkpoints/`.

## 8. Figures (regenerated ONLY from metrics.jsonl -- never from memory)

`make_figures(run_id)` produces, at minimum:
1. `training.png` -- loss + train/test acc vs epoch, vertical line at E_0.
2. `qnc1.png` -- QNC1_trace and QNC1_hs vs epoch, log-y, E_0 marked.
3. `qnc2.png` -- equinorm_cv and equiangle_dev vs epoch, log-y, horizontal
   line at 0, plus all pairwise cos_theta trajectories with etf_target dashed.
4. `entropy.png` -- entropy_mean and per-class entropy vs epoch.
5. `gram_triptych.png` -- cos_theta heatmaps at init / E_0 / final epoch.
6. `beta.png` -- beta trajectory.

All figures: seeds aggregated as mean line + shaded +/-1 std band when
multiple seeds exist for the config.

## 9. Numerical policy

- Eigen-decompositions: `numpy.linalg.eigh` (Hermitian) only.
- Clip eigenvalues to `[0, 1]` before entropy after asserting they exceed -1e-10.
- Seeds set for: numpy, torch (if used), PennyLane; recorded in JSONL.
- Tolerances above are hard test thresholds, not warnings.
