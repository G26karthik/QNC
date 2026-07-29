# RESULTS_V2.md — Quantum Neural Collapse: Phase 1 + Phase 2 Evidence Assembly

Assembled per PLAYBOOK2.md Stage 11 from `metrics.jsonl` logs under `results/`
(SPEC.md sections 7–8). All figures are pure functions of those JSONL files
(CLAUDE.md invariant 4). Every number in this document is copy-pasted from
`report.py` / `reanalyze.py` output as recorded in RESULTS.md, REANALYSIS.md,
RESULTS_STAGE9.md, or `results/sweeps/stage10_report.md` — no number has been
hand-computed or adjusted. This document reports what the numbers show and
nothing beyond them.

## Citations (mandatory per SPEC.md section 0.4 + SPEC2_ADDENDUM.md section 10)

- Papyan, Han, Donoho (2020), PNAS 117(40):24652–24663 — classical Neural
  Collapse definition (NC1–NC4).
- Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 — proof that the global
  optimizer of a quantum classifier under regularized squared-mean-error loss
  exhibits within-class concentration and mutually orthogonal class-mean
  measurement operators (quantum NC at the static optimum).
- San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 — empirical study of
  observable-set choices in multiclass quantum classification; confirms NC
  tendency across losses in measurement-operator space.
- Larocca, Ju, Garcia-Martin, Coles, Cerezo (2023), Nature Computational
  Science 3, 542–551 — theory of overparametrization in QNNs: spurious local
  minima disappear above a critical parameter count M_c, upper-bounded by the
  dimension of the ansatz's dynamical Lie algebra (DLA).
- Bowles, Ahmed, Schuld (2024), arXiv:2403.07059 — QML benchmark study;
  source of the bars-and-stripes and linearly-separable datasets and of the
  finding that entanglement is often dispensable for QML performance.
- Han, Papyan, Donoho (2022), ICLR — classical NC dynamics under MSE loss
  (precedent for the loss-arm comparison).

## 1. Methodology

**Model (SPEC.md section 2).** Angle-encoded VQC: inputs scaled to
`[-pi, pi]` per feature, applied as `RY(x_j)` with optional data re-uploading;
`qml.StronglyEntanglingLayers` ansatz with `L` layers and an `entangling`
on/off ablation switch; Pauli-Z readout on the first `C` qubits; learnable
inverse-temperature `beta` on the logits; exact statevector simulation
(`default.qubit`, `shots: null`, CLAUDE.md invariant 6). Phase 2 adds
amplitude encoding (`qml.AmplitudeEmbedding`, normalize=True) and an MSE loss
arm (SPEC2 section 14). Terminal Phase of Training (TPT) per SPEC.md
section 3: `E_0` = first epoch at 100% train accuracy; runs that never reach
it are `tpt_reached: false` and invalid for NC conclusions.

**Phase 1 witnesses (SPEC.md section 4).** QNC1 via mean within-class trace
distance (`qnc1_trace`) and a Hilbert–Schmidt analogue (`qnc1_hs`); QNC2 via
`equinorm_cv` and `equiangle_dev` against the simplex-ETF target
`-1/(C-1)`; entanglement via von Neumann entropy of the reduced state at a
fixed bipartition (`bipartition: first_half`).

**Phase 2 normalized witnesses (SPEC2 section 11).** Motivated by
concentration of measure: random pure states in dimension 2^n are nearly
maximally trace-distant, so the unnormalized `qnc1_trace` has almost no
dynamic range near its ceiling. New witnesses: `qnc1_fisher` = S_W/S_B
(within/between Hilbert–Schmidt ratio, the direct analog of classical NC1),
`qnc1_ratio` = D_within/D_between in trace distance, `overlap_offdiag` =
mean off-diagonal class-mean overlap Tr(rho_bar_c rho_bar_c') (the direct
test of Du et al.'s orthogonality prediction), and per-class purity.

**Overparameterization sweep (SPEC2 section 12).** Primary architecture
n = 4 qubits (Hilbert dimension 16, su(16) dimension 255). SEL gives
M = 12L parameters; DLA upper bound M_c <= 4^n - 1 = 255. Sweep
L in {2, 4, 8, 12, 16, 21, 26, 32} (M in {24 … 384}), spanning
M/M_c_bound ~ 0.09 to ~1.51, 5 seeds per L. Empirical capacity locator:
rank of the quantum Fisher information (quantum geometric tensor) matrix
averaged over 16 training inputs, logged at epoch 0 and final epoch.

**Datasets (SPEC2 section 13).** Phase 2 primary is `blobs3` (3 Gaussian
blobs in R^4, centers at distance 3.0, std 0.8, 60/class), chosen so
interpolation difficulty is controlled and the sweep measures landscape
effects. Generalization arms: `bars_stripes_4x4` (angle and amplitude
encoding), `linearly_separable_4d` (both from Bowles et al. 2024), and
`mnist_pca4` retained only as a secondary external-validity arm.

**Statistics.** All two-sample comparisons are Mann-Whitney U tests on
final-epoch values, pooling seeds. At n=3 per arm the test's floor p-value
is 0.1.

## 2. Phase 1 summary (full detail in RESULTS.md)

The Phase 1 campaign (Stages 1–5, MNIST PCA encodings, n=6 qubits primary)
established:

1. **Instrument validation.** A classical MLP control reproduced the known
   classical NC phenomenon (NC1 decay of ~1.3–1.4 orders of magnitude,
   pairwise cosines converging to the C=3 ETF target of −0.5) before any
   quantum measurement was made (RESULTS.md section 3).
2. **C=3 never interpolated.** The planned C=3 anchor (n=6, L=4, M=72)
   failed to reach TPT across 17 tuning attempts (best 98.89% train acc),
   while classical SVC/RF reached 98.9–100% on identical PCA-6 features;
   a C=2 diagnostic interpolated cleanly and became the Phase 1 anchor
   (RESULTS.md section 2). In Larocca et al. (2023)'s terms, M = 72 is
   ~1.8% of the n=6 DLA bound 4^6 − 1 = 4095.
3. **Null headline.** Post-TPT `qnc1_trace` at the C=2 anchor
   (0.945805 ± 0.00593142, 5 seeds) was statistically indistinguishable
   from a 10-seed untrained random-init null (0.946773 ± 0.00629237)
   (RESULTS.md sections 4, 6.2).
4. **Entanglement ablation.** Removing entangling gates dropped
   `tpt_reached` from 3/3 to 1/3 at the C=2 anchor and forced
   `entropy_mean` to ~8.5e-15 (product states), the largest effect size of
   any Phase 1 comparison (RESULTS.md sections 5.1, 6.4).

## 3. Stage 7 — Normalized-witness reanalysis of Phase 1 data

Question: was the Phase 1 null (point 3 above) an artifact of the
unnormalized trace-distance metric saturating near its random-state ceiling?
Method (REANALYSIS.md): `qnc1_fisher` recomputed from already-logged JSONL
fields for every Phase 1 run; `qnc1_ratio`, `overlap_offdiag`, `purity_mean`
recomputed by reloading checkpoints (trained arm) or reconstructing seeded
pre-training states (null arm) and regenerating states. No new training.

| metric | trained (C=2 anchor, final epoch, n=5) | null (random-init, n=10) | Mann-Whitney U p-value |
|---|---|---|---|
| qnc1_trace | 0.945805 +/- 0.00593142 | 0.946773 +/- 0.00629237 | 0.6787 |
| qnc1_fisher | 47.6565 +/- 7.7181 | 49.7667 +/- 8.16969 | 0.5941 |
| qnc1_ratio | 1.36352 +/- 0.0514688 | 1.37748 +/- 0.0667624 | 0.4396 |
| purity_mean | 0.0557393 +/- 0.0064114 | 0.0546314 +/- 0.00666157 | 0.6787 |

**Outcome: the reanalysis confirmed the Phase 1 null rather than overturning
it.** No normalized witness separates trained from untrained states at this
C=2 anchor (all p >= 0.44). The null is a property of this configuration,
not of the metric.

**Figures:**
- `results/stage2_vqc_c2_diagnostic_aggregate/figures/normalized_witnesses.png`
- `results/stage7_reanalysis/figures/four_witness_comparison.png`

## 4. Stage 9 — The overparameterization transition (headline result)

40/40 runs completed (8 L-values × 5 seeds), n=4, C=3, blobs3, CE loss,
600 epochs. Headline figure:
`results/sweeps/transition_L_order_parameter.png` (all outcome variables vs
M/M_c_bound on a shared x-axis). Full table in RESULTS_STAGE9.md.

| L | M | M/M_c_bound | TPT fraction | E0 | qnc1_fisher | qnc1_ratio | purity_mean | equiangle_dev | qfi_rank(final) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 24 | 0.094 | 0/5 | n/a | 10.3201 +/- 2.16272 | 1.28614 +/- 0.108321 | 0.183854 +/- 0.0209346 | 0.0693134 +/- 0.0221399 | 24 +/- 0 |
| 4 | 48 | 0.188 | 3/5 | 222.333 +/- 108.149 | 19.3549 +/- 2.83242 | 1.64869 +/- 0.0816001 | 0.129365 +/- 0.00572579 | 0.0340069 +/- 0.0196564 | 48 +/- 0 |
| 8 | 96 | 0.376 | 5/5 | 35.2 +/- 5.94643 | 29.7905 +/- 4.09096 | 1.91926 +/- 0.101013 | 0.105038 +/- 0.00637682 | 0.0327978 +/- 0.0181343 | 96 +/- 0 |
| 12 | 144 | 0.565 | 5/5 | 20.6 +/- 4.02989 | 32.0258 +/- 1.37397 | 1.97825 +/- 0.0414054 | 0.0999712 +/- 0.00159647 | 0.0259928 +/- 0.012871 | 144 +/- 0 |
| 16 | 192 | 0.753 | 5/5 | 12.4 +/- 1.62481 | 30.8934 +/- 2.52113 | 1.94291 +/- 0.0639831 | 0.100148 +/- 0.00266308 | 0.0348843 +/- 0.0160153 | 192 +/- 0 |
| 21 | 252 | 0.988 | 5/5 | 10.4 +/- 0.489898 | 28.3037 +/- 0.925056 | 1.88355 +/- 0.0410248 | 0.102459 +/- 0.00166004 | 0.0366689 +/- 0.00923561 | 252 +/- 0 |
| 26 | 312 | 1.224 | 5/5 | 8.2 +/- 0.979796 | 25.1575 +/- 1.02225 | 1.78765 +/- 0.0225087 | 0.106091 +/- 0.00180329 | 0.0295851 +/- 0.012523 | 311.8 +/- 0.4 |
| 32 | 384 | 1.506 | 5/5 | 8.4 +/- 0.8 | 22.5462 +/- 1.20778 | 1.72139 +/- 0.0379499 | 0.110409 +/- 0.00273857 | 0.0357493 +/- 0.0152925 | 384 +/- 0 |

What the numbers show:

- **TPT reachability transitions sharply and early.** 0/5 at L=2
  (M/M_c_bound = 0.094), 3/5 at L=4 (0.188), 5/5 from L=8 (0.376) onward.
  Where TPT is reached, E0 decreases monotonically with M
  (222.3 → 8.2 epochs).
- **The transition sits far below the DLA upper bound.** Full TPT
  reachability arrives by M = 96 ≈ 0.38 × 255, consistent with 4^n − 1
  being an upper bound on M_c, not M_c itself.
- **qfi_rank does not locate a saturation point below M = 384.** The
  batch-aggregated QFI rank equals n_params at every L except L26
  (311.8 ± 0.4 vs 312), including L32 where 384 > 255. The
  batch-aggregated Gram matrix (averaged over 16 inputs, SPEC2
  section 12.2) is evidently not bounded by the single-orbit DLA dimension;
  the rank readout therefore did not provide the empirical M_c locator that
  SPEC2 section 12.2 predicted.
- **Collapse witnesses do not jump at the TPT transition.** Comparing
  underparameterized (L<=12, n=20) vs overparameterized (L>=21, n=15)
  groups: qnc1_fisher p = 0.7515, equiangle_dev p = 0.9601. `qnc1_fisher`
  rises from 10.3 (L=2) to a maximum ~32.0 (L=12) then declines to
  22.5 (L=32); note higher qnc1_fisher = larger within/between ratio.
  `purity_mean` improves from 0.184 to ~0.10 and then flattens.
  In the terminology of the PLAYBOOK2 Stage 9 decision rule, the TPT jump
  is sharp but the collapse-witness response is gradual: the witnesses
  track capacity rather than jumping at the transition.

## 5. Stage 10 — Generalization arms

Arm selection from the Stage 9 table: **L_under = 2** (largest L with TPT
fraction ~0), **L_over = 8** (smallest L with TPT fraction ~1). Full tables
in `results/sweeps/stage10_report.md`.

### 5.1 Dataset generality at L_over = 8 (3 seeds each)

| variant | C | TPT fraction | E0 | qnc1_fisher | overlap_offdiag | purity_mean | equiangle_dev |
|---|---|---|---|---|---|---|---|
| bs4x4_angle | 2 | 3/3 | 12.6667 +/- 3.39935 | 47.666 +/- 5.19305 | 0.0575009 +/- 0.00167939 | 0.0957375 +/- 0.00262896 | ~0 (trivial, see note) |
| bs4x4_amplitude | 2 | 3/3 | 13.6667 +/- 0.471405 | 57.1988 +/- 2.53018 | 0.0563276 +/- 0.000674241 | 0.088152 +/- 0.00130965 | ~0 (trivial, see note) |
| linsep4d | 2 | 3/3 | 13.6667 +/- 1.24722 | 47.843 +/- 0.340202 | 0.0559515 +/- 0.00052971 | 0.0936822 +/- 0.000671025 | ~0 (trivial, see note) |
| mnist_pca4 | 3 | 3/3 | 21.3333 +/- 0.471405 | 14.6101 +/- 2.10462 | 0.0588152 +/- 0.000934791 | 0.147923 +/- 0.0131193 | 0.276058 +/- 0.0204106 |

TPT fraction is 3/3 for every dataset variant at L=8, including both
encodings of bars-and-stripes and the retained mnist_pca4 arm.
**Note on equiangle_dev:** with C=2 there is a single pairwise class-mean
angle, so the deviation across pairs is zero by construction (values
~1e-16). It is not evidence of collapse for the three C=2 variants and is
reported only for mnist_pca4 (C=3). qfi_rank(final) = 96 = n_params for all
variants, consistent with Stage 9 at L=8.

**Figure:** `results/sweeps/stage10_datasets_20260713-104835/comparison.png`

### 5.2 CE vs MSE at L_over = 8, blobs3, matched 5 seeds

**Framing (this is the direct comparison to Du et al.'s setting):**
Du et al. (2023)'s theorem characterizes the *global optimizer* of the
*regularized-MSE* risk. Our MSE runs did not reach interpolation under the
CE-tuned 600-epoch budget (0/5 TPT; final train accuracies 0.8750–0.9306;
all runs ended at epoch 600), while CE reached E0 = 35.2 ± 5.9 on the
identical configuration (5/5 TPT). The two arms therefore compare **training
dynamics at matched budget, not optimum-vs-optimum**; the MSE column below
describes a non-interpolating trajectory, not the optimum Du et al.'s
theorem describes.

| witness | ce (mean+/-std) | mse (mean+/-std) | Mann-Whitney U p-value |
|---|---|---|---|
| tpt_fraction | 5/5 | 0/5 | -- |
| E0 | 35.2 +/- 5.94643 | n/a | -- |
| qnc1_trace | 0.898221 +/- 0.00557337 | 0.897896 +/- 0.00351256 | 0.8413 |
| qnc1_fisher | 29.7905 +/- 4.09096 | 31.8184 +/- 3.53008 | 0.4206 |
| qnc1_ratio | 1.91926 +/- 0.101013 | 2.01659 +/- 0.0797679 | 0.2222 |
| overlap_offdiag | 0.0591396 +/- 0.000482845 | 0.0627548 +/- 0.00118868 | 0.007937 |
| purity_mean | 0.105038 +/- 0.00637682 | 0.105428 +/- 0.00392964 | 0.8413 |
| equiangle_dev | 0.0327978 +/- 0.0181343 | 0.0401358 +/- 0.0174529 | 0.8413 |
| entropy_mean | 1.32433 +/- 0.0129475 | 1.32975 +/- 0.0201296 | 0.6905 |

All witnesses are statistically indistinguishable between the arms
(p >= 0.22) except `overlap_offdiag` (p = 0.0079; CE lower by ~6%).
Candidate diagnoses for the MSE non-interpolation, recorded but not tested
(stage10_report.md): (a) MSE gradient scale near one-hot targets is smaller
than CE's, so the shared lr/epoch budget under-trains the MSE arm;
(b) beta-ceiling interaction with the MSE objective.

**Figure:** `results/sweeps/stage10_loss_20260713-113927/loss_comparison.png`

### 5.3 Entanglement × transition (ablation with corrected DLA bounds)

Removing entanglers changes the DLA: the non-entangling ansatz generates
su(2)^{⊗n}, dimension 3n = 12, **regardless of L**. It is therefore always
effectively parameter-saturated and can never exceed su(2)^{⊗n} capacity.
`m_over_mc` for the ablation arms is logged against this corrected bound
(12), not 4^n − 1 = 255: 24/12 = 2.0 at L=2 and 96/12 = 8.0 at L=8.

| arm | M | m_over_mc (bound used) | TPT fraction | qnc1_fisher | purity_mean | entropy_mean | qfi_rank(final) |
|---|---|---|---|---|---|---|---|
| L2, entangling | 24 | 0.094 (255) | 0/5 | 10.3201 +/- 2.16272 | 0.183854 +/- 0.0209346 | 1.34851 +/- 0.0556213 | 24 +/- 0 |
| L2, non-entangling | 24 | 2.0 (12) | 0/3 | 3.92245 +/- 1.05237 | 0.421579 +/- 0.0661909 | 5.42731e-15 +/- 3.15945e-17 | 20 +/- 0 |
| L8, entangling | 96 | 0.376 (255) | 5/5 | 29.7905 +/- 4.09096 | 0.105038 +/- 0.00637682 | 1.32433 +/- 0.0129475 | 96 +/- 0 |
| L8, non-entangling | 96 | 8.0 (12) | 0/3 | 5.16621 +/- 0.660428 | 0.272412 +/- 0.044003 | 4.87365e-15 +/- 9.40163e-16 | 66 +/- 0.816497 |

- The non-entangling ansatz reaches TPT in 0/3 seeds at **both** L=2 and
  L=8 — including at the parameter count (96) where the entangling ansatz
  reaches 5/5. Additional parameters do not rescue the non-entangling
  ansatz on this task.
- `entropy_mean` ~5e-15 for the non-entangling arms (zero to numerical
  precision): the circuit produces product states, confirming the ablation
  removes what it claims to remove.
- The Phase 1 entanglement finding (RESULTS.md section 5.1) survives the
  overparameterization reframing and gains a capacity explanation: the
  non-entangling ansatz has a DLA capacity ceiling of 12 that no amount of
  depth can raise.
- qfi_rank(final) for the non-entangling arms (20 at L=2; 66 ± 0.8 at L=8)
  is the rank of the batch-aggregated parameter-space QFI Gram matrix
  (SPEC2 section 12.2), which is not upper-bounded by the single-orbit DLA
  dimension; it is reported as logged, without further interpretation.

**Figure:** `results/sweeps/stage10_ablation_20260713-115624/ablation_bounds.png`

## 6. Relation to prior work

### 6.1 Du, Yang, Tao, Hsieh (2023), PRL 131, 140601

**Their result:** at the global optimizer of the regularized squared-error
risk, feature states concentrate around per-class means and class-mean
measurement operators become mutually orthogonal; this structure is
equivalent to classical NC at the optimum.

**Our evidence:** the `overlap_offdiag` witness is the direct test of their
orthogonality prediction in density-matrix space. In interpolating CE runs
at L=8 it sits at 0.0591 ± 0.0005 (not at 0). Our MSE arm — the loss their
theorem covers — did not reach the optimum their theorem describes within
the matched 600-epoch budget (0/5 TPT, train acc 0.875–0.931), so this
document contains **no optimum-vs-optimum comparison to their theorem**;
what it shows is that at matched budget the CE and MSE trajectories are
witness-indistinguishable (p >= 0.22) except for overlap_offdiag
(p = 0.0079). A budget-matched-at-interpolation MSE comparison remains
future work.

### 6.2 Larocca, Ju, Garcia-Martin, Coles, Cerezo (2023), Nat. Comput. Sci.

**Their result:** underparametrized QNNs suffer spurious local minima that
disappear above a critical M_c, upper-bounded by the DLA dimension; capacity
saturates when QFI rank stops growing with M.

**Our evidence:** TPT reachability does jump sharply with M (0/5 → 3/5 →
5/5 across M = 24, 48, 96), qualitatively consistent with a
trainability transition, and the Phase 1 C=3 failure at M = 72 (~1.8% of
the n=6 bound) is consistent with their underparametrized regime. The jump
occurs at M/M_c_bound ≈ 0.19–0.38, well below the su(2^n) bound — as
expected for an upper bound. However, our batch-aggregated QFI rank equaled
n_params at nearly every L (including M = 384 > 255) and thus did not
exhibit the saturation their single-state criterion predicts; our
implementation averages the geometric tensor over 16 inputs, which is not
bounded by the single-orbit DLA dimension. The ablation result (section
5.3) matches their framework's prediction that a DLA of dimension 12 caps
capability regardless of parameter count.

### 6.3 Bowles, Ahmed, Schuld (2024), arXiv:2403.07059

**Their result:** across a broad benchmark, entanglement in the model is
often dispensable — non-entangling models frequently match entangling ones.

**Our evidence is in tension on the specific axis of interpolation:** on
blobs3 at n=4, the non-entangling ansatz reached TPT in 0/3 seeds at both
tested depths while the entangling ansatz reached 5/5 at the same parameter
count (96). We measure exact interpolation (100% train accuracy), not test
accuracy — Bowles et al.'s dispensability claim concerns generalization
performance, so the two findings address different quantities and can
coexist. The DLA capacity argument (ceiling 12 for product ansatze at n=4)
gives a mechanism for why interpolation of this C=3 task fails without
entanglers.

### 6.4 San Sebastian, Canizo, Orus (2026), arXiv:2602.08485

**Their result:** empirical NC tendency in quantum classifiers across losses
and observable sets, in measurement-operator space.

**Our evidence:** complementary axis — we work in density-matrix space
(trace distance, HS ratios, class-mean overlap, purity) and study dynamics,
entanglement ablation, and the overparameterization transition. Our
CE-vs-MSE result (witnesses indistinguishable at matched budget except
overlap_offdiag) is consistent with their across-loss NC tendency, with the
budget caveat of section 5.2.

## 7. Limitations

1. **Simulator-only.** All Phase 2 runs use exact statevector simulation
   (`default.qubit`, `shots: null`). No claim transfers to hardware noise
   (readout error, decoherence, gate infidelity). The single finite-shot
   control is Phase 1's `stage5_shots` (training-gradient noise only).
2. **n = 4 primary.** The transition sweep and all Stage 10 arms run at 4
   qubits. Phase 1 touched n = 4/6/8 only in a small non-interpolating
   sweep. Whether the transition location scales with n is untested.
3. **Single bipartition.** `entropy_mean` uses one fixed cut
   (`first_half`); entanglement structure across other cuts is unmeasured.
4. **The DLA bound is an upper bound, not M_c.** 4^n − 1 = 255 bounds M_c
   from above; the observed TPT transition at M ≈ 48–96 does not locate
   M_c, and our batch-aggregated QFI rank did not provide the intended
   empirical locator (section 4). No exact M_c is reported anywhere in this
   document.
5. **blobs3 is synthetic.** The headline sweep uses a controlled Gaussian
   dataset by design (landscape effects, not data pathology). Real-data
   external validity at L_over rests on the smaller (3-seed) Stage 10
   dataset arms, of which mnist_pca4 is the only non-synthetic one.
6. **Carried over from Phase 1:** the Phase 1 trained-vs-null comparison is
   at a C=2 anchor where `equiangle_dev` is trivial; the MSE arm's
   non-interpolation (section 5.2) means the Du-theorem comparison is
   budget-matched dynamics only; C=2 dataset variants report no meaningful
   `equiangle_dev`.

## 8. Run traceability per figure

All run directories live under `results/`; per-variant run_id maps are in
the named manifest JSONs.

| Figure | Source runs |
|---|---|
| `results/sweeps/transition_L_order_parameter.png` | 40 runs in `results/sweeps/transition_L_manifest.json`: `transition_L_L{2,4,8,12,16,21,26,32}_s{0..4}_*` (L2: s0 20260712-173147, s1 173447, s2 173630, s3 173816, s4 173957; L4: s0 20260712-171813, s1 193008, s2 193432, s3 193755, s4 194505; L8: s0 194651, s1 194946, s2 195241, s3 195535, s4 195831; L12: s0 200127, s1 200546, s2 201003, s3 201421, s4 201838; L16: s0 202254, s1 202837, s2 203418, s3 204000, s4 204542; L21: s0 205124, s1 205847, s2 210603, s3 211942, s4 213519; L26: s0 215037, s1 220816, s2 222145, s3 223419, s4 224802; L32: s0 225736, s1 230804, s2 231832, s3 232902, s4 233938 — all 20260712) |
| `results/sweeps/stage10_datasets_20260713-104835/comparison.png` | 12 runs in that dir's `manifest.json`: `stage10_datasets_bs4x4_angle_s{0,1,2}_20260713-{061018,061305,061549}`, `stage10_datasets_bs4x4_amplitude_s{0,1,2}_20260713-{061841,064311,065646}`, `stage10_datasets_linsep4d_s{0,1,2}_20260713-{071401,071805,072051}`, `stage10_datasets_mnist_pca4_s{0,1,2}_20260713-{103223,103902,104431}` |
| `results/sweeps/stage10_loss_20260713-113927/loss_comparison.png` (and `comparison.png`) | 10 runs in that dir's `manifest.json`: `stage10_loss_ce_s{0..4}_20260713-{104836,105340,105725,110109,110452}`, `stage10_loss_mse_s{0..4}_20260713-{110836,111319,111954,112633,113332}` |
| `results/sweeps/stage10_ablation_20260713-115624/ablation_bounds.png` (and `comparison.png`) | 6 ablation runs in that dir's `manifest.json`: `stage10_ablation_noent_L2_s{0,1,2}_20260713-{113928,114053,114214}`, `stage10_ablation_noent_L8_s{0,1,2}_20260713-{114335,114743,115204}`; entangling comparators are the Stage 9 L2/L8 runs above |
| `results/stage7_reanalysis/figures/four_witness_comparison.png` | trained arm: `stage2_vqc_c2_diagnostic_s{0..4}` (s0 20260710-002127, s1 110536, s2 111510, s3 113205, s4 114140); null arm: `stage5_random_init_s0_20260710-144603` … `s9_20260710-144816` |
| `results/stage2_vqc_c2_diagnostic_aggregate/figures/normalized_witnesses.png` | same 5 trained runs as above |
| Phase 1 figures (Stage 1–5) | traced in RESULTS.md sections 3–6; manifests: `ablation_entangling_20260710-122347`, `sweep_classes_20260710-123400`, `sweep_depth_20260710-123425`, `sweep_qubits_20260710-133607`, `sweep_wd_20260710-123448`, `stage5_shuffled_labels_20260710-153130`, `stage5_shots_20260712-060047` |

Orphaned partial run dirs from externally killed Stage 10 driver processes
are kept (append-only, CLAUDE.md invariant 3) but excluded from every
manifest and table: `stage10_datasets_bs4x4_amplitude_s1_20260713-062934`,
`stage10_datasets_mnist_pca4_s0_20260713-072337`,
`stage10_datasets_mnist_pca4_s0_20260713-072637`.
