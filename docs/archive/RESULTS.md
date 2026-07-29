# RESULTS.md — Quantum Neural Collapse: Training Dynamics

Assembled from `metrics.jsonl` logs under `results/` per SPEC.md section 7/8.
All figures are pure functions of those JSONL files (never hand-edited). All
numbers below are copy-pasted from `report.py` output or `results/stage5_statistics.md`
— no number in this document has been hand-computed or adjusted.

## 1. Methodology

**Model.** SPEC.md section 2: angle-encoded VQC (PCA-projected features scaled to
`[-pi, pi]`), `StronglyEntanglingLayers` ansatz with an `entangling` on/off switch,
Pauli-Z readout on the first `C` qubits, learnable logit temperature `beta`,
exact statevector simulation (`default.qubit`, `shots: null` by default per
CLAUDE.md invariant 6). Terminal Phase of Training (TPT) is defined per
SPEC.md section 3.

**Metrics.** SPEC.md section 4: QNC1 (within-class concentration) via two
independent witnesses — mean pairwise trace distance and a Hilbert–Schmidt
analogue — computed on density matrices `rho_i` built from `qml.state`. QNC2
(class-mean geometry) via `equinorm_cv` and `equiangle_dev` against the
Simplex-ETF target angle (`etf_target = -1/(C-1)`). Entanglement entropy via
von Neumann entropy of the reduced density matrix at a fixed bipartition
(`bipartition: first_half`). Classical NC1/NC2 (SPEC.md section 5) on MLP
penultimate features serve as the Stage 1 instrument-validation control.

**Contribution framing (SPEC.md section 0.3).** Du et al. (2023) proved the
*static* optimum of quantum classifiers exhibits orthogonal/ETF structure.
This project does not re-derive that result; it measures the **training
trajectory** toward (or away from) it, using trace distance as the primary
witness, and runs a controlled entanglement ablation to test whether that
trajectory requires entanglement. Every figure/table below is a factual
trajectory report, not a claim of collapse or its absence beyond what the
numbers show.

**Mandatory citations (SPEC.md section 0.4 / CLAUDE.md):**
- Papyan, Han, Donoho (2020), PNAS 117(40):24652–24663 — classical NC definition
- Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 — quantum NC at the static optimum
- San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 — empirical multi-loss/observable-set QNC study

## 2. Anchor-config decision (context for every table below)

The originally planned Stage 2 anchor (`stage2_vqc_base.yaml`, MNIST digits
0/1/2, C=3, n=6 qubits PCA encoding) never reached `tpt_reached=true` across
17 tuning attempts (lr, beta0, qubit count, optimizer incl. L-BFGS,
hardware-efficient ansatz) — best result 98.89% train acc, short of exact
interpolation. Root-cause checks ruled out a data/PCA bottleneck (classical
SVC/RF hit 98.9–100% on identical PCA-6 features). A C=2 diagnostic
(`stage2_vqc_c2_diagnostic.yaml`, digits 0/1, otherwise identical
architecture) reaches 100% train acc cleanly and was adopted as the anchor
for Stage 3 (5-seed dynamics campaign) and the Stage 4/5 sweeps that hold
architecture fixed (`ablation_entangling`, `stage5_shots`,
`stage5_shuffled_labels`). `sweep_classes`, `sweep_depth`, `sweep_qubits`,
`sweep_wd` instead sweep the C=3 base architecture directly (see section 4) —
those rows report the swept variable's effect on `tpt_reached` itself, which
is scientifically informative in its own right.

## 3. Stage 1 — Classical NC control (instrument validation)

Run IDs: `stage1_classical_s0_20260709-225717`,
`stage1_classical_s1_20260709-225806`, `stage1_classical_s2_20260709-225849`
(MLP 784-512-512-64-C, MNIST digits {3,5,8}, C=3, 500/class, 1500 epochs).

100% train acc reached and held for ≥5x past E0 in all 3 seeds. NC1 decayed
~1.3–1.4 orders of magnitude from its early-training peak (~1.3–1.8 at init
to a ~0.05–0.07 floor); `equiangle_dev` and `equinorm_cv` converged toward
0.03; pairwise cosines converged toward the C=3 ETF target of −0.5. Digits
{3,5,8} were used instead of {0,1,2} because the latter are near-linearly
separable in raw pixels, compressing NC1's dynamic range at init. This
confirms the `metrics.py` instrument reproduces the known classical NC
phenomenon before any quantum claim is made.

**Figure:** `results/stage1_classical_s0_20260709-225717/figures/qnc1.png`

## 4. Stage 3 — Quantum NC training-dynamics campaign (headline result)

Anchor: `stage2_vqc_c2_diagnostic.yaml` (C=2, digits 0/1, n=6 qubits, L=4,
reupload, entangling, PCA-6, 400 epochs), 5 seeds.

Run IDs: `stage2_vqc_c2_diagnostic_s0_20260710-002127`,
`stage2_vqc_c2_diagnostic_s1_20260710-110536`,
`stage2_vqc_c2_diagnostic_s2_20260710-111510`,
`stage2_vqc_c2_diagnostic_s3_20260710-113205`,
`stage2_vqc_c2_diagnostic_s4_20260710-114140`.
Aggregate: `results/stage2_vqc_c2_diagnostic_aggregate/`.

| seed | E0 | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|---|
| 0 | 31 | 0.941089 | 0.939339 | 0 | 1.11022e-16 | 2.26095 |
| 1 | 104 | 0.936801 | 0.934412 | 0.03125 | 2.22045e-16 | 2.26933 |
| 2 | 142 | 0.947835 | 0.946425 | 0.03125 | 5.55112e-16 | 2.27753 |
| 3 | 27 | 0.951936 | 0.95085 | 1.51251e-16 | 2.22045e-16 | 2.25852 |
| 4 | 115 | 0.951363 | 0.950278 | 0.01875 | 2.22045e-16 | 2.29852 |
| **mean+/-std** | -- | 0.945805 +/- 0.00593142 | 0.944261 +/- 0.0064114 | 0.01625 +/- 0.0140312 | 2.66454e-16 +/- 1.50598e-16 | 2.27297 +/- 0.0144234 |

5/5 seeds reach `tpt_reached=true` (E0 ranges 27–142 epochs). Reported
`qnc1_trace`/`qnc1_hs` values are the **final-epoch** readings (post-TPT), not
the peak — they sit near ~0.94–0.95, essentially unchanged from the
random-init null distribution (section 6.2 below) at this C=2, balanced-class
setting where `equiangle_dev` is trivially ~0 by construction (centering
forces `M_1 = -M_0` exactly). `equinorm_cv` stays near 0–0.03 across seeds and
epochs. Entanglement entropy (`entropy_mean`) stays near 2.25–2.30 across all
5 seeds post-TPT.

**Figures:**
- `results/stage2_vqc_c2_diagnostic_aggregate/figures/training.png` (mean ± 1 std, 5 seeds)
- `results/stage2_vqc_c2_diagnostic_aggregate/figures/qnc1.png`
- `results/stage2_vqc_c2_diagnostic_aggregate/figures/qnc2.png`
- `results/stage2_vqc_c2_diagnostic_aggregate/figures/entropy.png`
- `results/stage2_vqc_c2_diagnostic_s0_20260710-002127/figures/gram_triptych.png` (init / E0 / final, seed 0)

## 5. Stage 4 — Cross-config sweeps

### 5.1 Entanglement ablation (key scientific ablation)

Manifest: `results/sweeps/ablation_entangling_20260710-122347/manifest.json`

| variant | n_seeds | tpt_reached | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|---|---|
| entangling_true | 3 | 3/3 | 0.941908 +/- 0.00454142 | 0.940058 +/- 0.00493038 | 0.0208333 +/- 0.0147314 | 2.96059e-16 +/- 1.88702e-16 | 2.26927 +/- 0.0067694 |
| entangling_false | 3 | 1/3 | 0.868834 +/- 0.0458841 | 0.859701 +/- 0.0534227 | 0.0208333 +/- 0.0147314 | 3.70074e-17 +/- 5.23364e-17 | 8.47735e-15 +/- 7.85278e-17 |

With entanglement disabled, `tpt_reached` drops from 3/3 to 1/3, `qnc1_trace`
mean drops from 0.9419 to 0.8688 with a much larger std (0.0459 vs 0.0045),
and `entropy_mean` collapses to ~8.5e-15 (i.e. exactly 0, as expected for a
non-entangling ansatz — a sanity check that entropy is measuring what it
claims to). `equinorm_cv` is unchanged between arms.

**Figure:** `results/sweeps/ablation_entangling_20260710-122347/comparison.png`

### 5.2 Class count (C=2,3,4)

Manifest: `results/sweeps/sweep_classes_20260710-123400/manifest.json`

| variant | n_seeds | tpt_reached | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|---|---|
| C2 | 3 | 3/3 | 0.94438 +/- 0.00376129 | 0.942917 +/- 0.00399765 | 0.0194444 +/- 0.0103935 | 1.4803e-16 +/- 1.04673e-16 | 2.25196 +/- 0.0113994 |
| C3 | 3 | 0/3 | 0.932449 +/- 0.00734682 | 0.929616 +/- 0.00814595 | 0.222791 +/- 0.0220375 | 0.416618 +/- 0.0590153 | 2.25246 +/- 0.00561642 |
| C4 | 3 | 0/3 | 0.939503 +/- 0.00354524 | 0.937027 +/- 0.00409844 | 0.289364 +/- 0.0463815 | 0.35138 +/- 0.0827219 | 2.25451 +/- 0.018416 |

Only C=2 reaches `tpt_reached=3/3` at this architecture/sample budget; C=3
and C=4 reach 0/3, consistent with the Stage 2 finding that the C=3 base
architecture never interpolates. `equiangle_dev` for C=3/C=4 (0.42, 0.35) is
far from the respective ETF targets (−0.5, −0.333 → the deviation metric is
not yet near 0), reflecting the non-interpolating regime.

**Figure:** `results/sweeps/sweep_classes_20260710-123400/comparison.png`

### 5.3 Circuit depth (L=2,4,6)

Manifest: `results/sweeps/sweep_depth_20260710-123425/manifest.json`

| variant | n_seeds | tpt_reached | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|---|---|
| L2 | 3 | 0/3 | 0.88999 +/- 0.0123665 | 0.883019 +/- 0.0146318 | 0.189769 +/- 0.0188466 | 0.33539 +/- 0.0536418 | 2.09012 +/- 0.139446 |
| L4 | 3 | 0/3 | 0.932449 +/- 0.00734682 | 0.929616 +/- 0.00814595 | 0.222791 +/- 0.0220375 | 0.416618 +/- 0.0590153 | 2.25246 +/- 0.00561642 |
| L6 | 3 | 0/3 | 0.945984 +/- 0.00593 | 0.944182 +/- 0.00651807 | 0.224769 +/- 0.02264 | 0.422018 +/- 0.0615028 | 2.28008 +/- 0.00526959 |

None of L=2/4/6 reach TPT at the C=3 base architecture. `qnc1_trace` and
`entropy_mean` both increase monotonically with depth (0.890→0.946 and
2.09→2.28 respectively); `equiangle_dev` also increases with depth
(0.335→0.422), moving *away* from the C=3 ETF target of −1/3 as capacity
grows, in this non-interpolating regime.

**Figure:** `results/sweeps/sweep_depth_20260710-123425/comparison.png`

### 5.4 Qubit count (n=4,6,8)

Manifest: `results/sweeps/sweep_qubits_20260710-133607/manifest.json`

| variant | n_seeds | tpt_reached | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|---|---|
| n4 | 3 | 0/3 | 0.843826 +/- 0.00134955 | 0.834654 +/- 0.00133809 | 0.130016 +/- 0.0223942 | 0.19766 +/- 0.0455018 | 1.27763 +/- 0.0156692 |
| n6 | 3 | 0/3 | 0.932449 +/- 0.00734682 | 0.929616 +/- 0.00814595 | 0.222791 +/- 0.0220375 | 0.416618 +/- 0.0590153 | 2.25246 +/- 0.00561642 |
| n8 | 3 | 0/3 | 0.966923 +/- 0.00240988 | 0.96608 +/- 0.00258464 | 0.215281 +/- 0.0175715 | 0.396426 +/- 0.0299054 | 3.15906 +/- 0.0162137 |

None of n=4/6/8 reach TPT at the C=3 base architecture. `entropy_mean` scales
strongly with qubit count (1.28→2.25→3.16, roughly tracking `log2(n)`
bipartition capacity), and `qnc1_trace` increases with n (0.844→0.932→0.967).

**Figure:** `results/sweeps/sweep_qubits_20260710-133607/comparison.png`

### 5.5 Weight decay (0, 1e-4, 1e-3)

Manifest: `results/sweeps/sweep_wd_20260710-123448/manifest.json`

| variant | n_seeds | tpt_reached | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|---|---|
| wd0 | 3 | 0/3 | 0.932449 +/- 0.00734682 | 0.929616 +/- 0.00814595 | 0.222791 +/- 0.0220375 | 0.416618 +/- 0.0590153 | 2.25246 +/- 0.00561642 |
| wd1e-4 | 3 | 0/3 | 0.930921 +/- 0.00922788 | 0.927712 +/- 0.0105794 | 0.226113 +/- 0.0255079 | 0.425263 +/- 0.0662995 | 2.25026 +/- 0.0149689 |
| wd1e-3 | 3 | 0/3 | 0.932654 +/- 0.00646614 | 0.929966 +/- 0.00686135 | 0.22386 +/- 0.0191156 | 0.418775 +/- 0.0486142 | 2.26458 +/- 0.00575121 |

None of the three weight-decay settings reach TPT at the C=3 base
architecture, and all five metrics are within ~1 std of each other across
wd0/wd1e-4/wd1e-3 — weight decay has negligible effect at this non-interpolating
configuration, unlike its known strengthening effect on classical NC
(Stage 1).

**Figure:** `results/sweeps/sweep_wd_20260710-123448/comparison.png`

## 6. Stage 5 — Controls and statistics

Full source: `results/stage5_statistics.md`. All comparisons below use
Mann-Whitney U (n=3 per arm unless noted; floor p-value 0.1 at n=3).

### 6.1 Shots noise (exact vs. 1000-shot training)

Manifest: `results/sweeps/stage5_shots_20260712-060047/manifest.json`. Metrics
are always computed from the exact analytic state (SPEC.md section 4); `shots`
affects training only, via parameter-shift gradient noise.

| metric | shots_null | shots_1000 | p |
|---|---|---|---|
| qnc1_trace | 0.941908 +/- 0.00454142 | 0.942528 +/- 0.00481697 | 1 |
| qnc1_hs | 0.940058 +/- 0.00493038 | 0.940869 +/- 0.00519088 | 1 |
| equinorm_cv | 0.0208333 +/- 0.0147314 | 0.0208333 +/- 0.0147314 | 0.5066 |
| equiangle_dev | 2.96059e-16 +/- 1.88702e-16 | 1.4803e-16 +/- 1.04673e-16 | 0.6428 |
| entropy_mean | 2.26927 +/- 0.0067694 | 2.2164 +/- 0.047677 | 0.1 |

No metric differs significantly (all p ≥ 0.1); the final geometry is not
disrupted by 1000-shot noisy parameter-shift training at this anchor config.

### 6.2 Random-init null distribution (10 seeds, 0 training epochs)

Config: `configs/stage5_random_init.yaml`. Run IDs:
`stage5_random_init_s0_20260710-144603` through
`stage5_random_init_s9_20260710-144816`.

| | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |
|---|---|---|---|---|---|
| **mean+/-std** | 0.946773 +/- 0.00629237 | 0.945369 +/- 0.00666157 | 0.02 +/- 0.0152582 | 1.66533e-16 +/- 1.02358e-16 | 2.27508 +/- 0.0201859 |

`equiangle_dev` is trivially ~0 at C=2 by centering construction and is not a
learning signal at this C. Critically, the **untrained** `qnc1_trace` null
(0.9468 +/- 0.0063) is statistically indistinguishable from the **post-TPT
trained** value from the Stage 3 campaign (0.9458 +/- 0.0059, section 4) —
`qnc1_trace` does not separate trained from untrained states at this C=2
anchor. `entropy_mean` similarly sits within one std of the trained value
(2.275 vs 2.273).

### 6.3 Real vs. shuffled labels

Manifest: `results/sweeps/stage5_shuffled_labels_20260710-153130/manifest.json`

| metric | real_labels | shuffled_labels | p |
|---|---|---|---|
| qnc1_trace | 0.941908 +/- 0.00454142 | 0.959973 +/- 0.00205726 | 0.1 |
| qnc1_hs | 0.940058 +/- 0.00493038 | 0.95932 +/- 0.00211925 | 0.1 |
| equinorm_cv | 0.0208333 +/- 0.0147314 | 0.0208333 +/- 0.0147314 | 0.7 |
| equiangle_dev | 2.96059e-16 +/- 1.88702e-16 | 3.70074e-17 +/- 5.23364e-17 | 0.1157 |
| entropy_mean | 2.26927 +/- 0.0067694 | 2.29292 +/- 0.01518 | 0.2 |

Shuffled-label training reaches TPT with `qnc1_trace` mean 0.9600, slightly
*higher* (less concentrated) than real-label training (0.9419), at p=0.1
(n=3 floor — not below the conventional 0.05 significance threshold).
`equinorm_cv` is identical between arms.

### 6.4 Entangled vs. not (Stage 4 ablation, statistical view)

Manifest: `results/sweeps/ablation_entangling_20260710-122347/manifest.json`

| metric | entangling_true | entangling_false | p |
|---|---|---|---|
| qnc1_trace | 0.941908 +/- 0.00454142 | 0.868834 +/- 0.0458841 | 0.1 |
| qnc1_hs | 0.940058 +/- 0.00493038 | 0.859701 +/- 0.0534227 | 0.1 |
| equinorm_cv | 0.0208333 +/- 0.0147314 | 0.0208333 +/- 0.0147314 | 0.5066 |
| equiangle_dev | 2.96059e-16 +/- 1.88702e-16 | 3.70074e-17 +/- 5.23364e-17 | 0.1157 |
| entropy_mean | 2.26927 +/- 0.0067694 | 8.47735e-15 +/- 7.85278e-17 | 0.1 |

`entropy_mean` p=0.1 is the n=3 floor for a difference of ~2.27 vs ~0 —
the largest effect size of any comparison in this document.

## 7. Full test suite

`pytest -q`: 70 passed at the time of this document's assembly (all Stage 0–5
work committed through `d81b9d9 stage5: controls`; `stage5_shots` sweep
results generated after that commit, results-only, gitignored per CLAUDE.md).

## 8. Limitations

- **Simulator-only.** All runs use `default.qubit` exact statevector
  simulation by default (CLAUDE.md invariant 6); the one exception
  (`stage5_shots`, `shots=1000`) only injects noise into *training* gradients
  via parameter-shift — every logged metric in this document, including that
  run's, is still computed from the exact analytic state. No results here
  bear on real quantum hardware noise (readout error, decoherence, gate
  infidelity).
- **Small n.** 200 samples/class train, 200 epochs of dense logging at most;
  qubit counts tested are 4, 6, 8 only (`sweep_qubits`); the Stage 3 dynamics
  campaign anchor uses C=2 (2 classes) because the C=3 architecture never
  reached TPT (section 2) — the headline trajectory result is demonstrated
  at C=2 only, not the originally planned C=3 setting.
- **Single bipartition for entanglement entropy.** `entropy_mean` is computed
  at one fixed bipartition (`bipartition: first_half`, SPEC.md section 4.4)
  across all runs; no bipartition sweep was run, so entanglement structure
  across other cuts of the qubit register is unmeasured.
- **Single encoding family.** All runs use PCA-projected angle encoding into
  `StronglyEntanglingLayers` (ring-topology) with an optional reupload flag;
  the one architecture variant explored (hardware-efficient ansatz, RY+RZ +
  linear CNOT ladder) was tried only during Stage 2 tuning (best 98.89% train
  acc on C=3, never reaching TPT) and is not part of the Stage 3–5 metric
  campaigns.
- **Comparison to Du et al. (2023) is qualitative, not a fit.** Du et al.
  prove a property of the *global optimizer* under a regularized
  squared-mean-error loss and an unconstrained-features model; this project
  trains under cross-entropy with a learnable logit temperature and measures
  the empirical trajectory of a finite VQC under gradient descent. No formal
  distance-to-theoretical-optimum quantity was computed; comparisons in this
  document are limited to whether logged metrics move toward the values
  Du et al.'s optimum would imply (e.g. `equiangle_dev` → 0), not to a
  quantitative convergence-rate claim.
- **`tpt_reached` gates most quantum tables.** Sections 5.2–5.5 report metrics
  from runs that never reached `tpt_reached=true`; those numbers describe
  training-in-progress states, not the terminal phase, and are not directly
  comparable to the Stage 3 (C=2, TPT-reached) numbers in section 4.
