# STATUS.md — Orientation Capsule

Read this first, every session. It replaces the findings files as the
default entry point; open a findings file only when a task needs its
per-seed numbers or methodology detail.

## Project statement

This codebase studies the **training dynamics** of Neural Collapse in
Variational Quantum Circuits (VQCs), building on Du, Yang, Tao, Hsieh
(2023, PRL 131, 140601)'s proof that QNC holds at the *static optimum*.
Phase 3's headline finding is **"collapse by rotation"**: post-TPT VQCs
(without data re-uploading) collapse their measurement-visible z-space
statistics (NC1_z decays 2–3+ orders of magnitude) while their *total*
Hilbert-Schmidt within-class variance is conserved to machine precision.
Collapse is not a real contraction of the state constellation — it is a
training-driven **rotation** that redistributes a near-fixed variance
budget out of the loss-visible measurement subspace and into the
loss-invisible fiber directions. This dichotomy replicates at a second
qubit count (n=6) and strengthens with n as dimension-counting predicts.
Citations mandatory in any writeup: Papyan/Han/Donoho (2020, PNAS),
Du/Yang/Tao/Hsieh (2023, PRL 131, 140601), San Sebastian/Canizo/Orus
(2026, arXiv:2602.08485).

## Confirmed results

| Result | Magnitude | Stage | Findings file |
|---|---|---|---|
| Machine-precision total-variance invariance (reupload=false) | rel. change ~1e-15, 5/5 seeds (n=4, HR2); ~5e-15, 5/5 seeds (n=6, HN1) | 14B, 15 | STAGE14B_FINDINGS.md, STAGE15_FINDINGS.md |
| NC1_z decay ≥1 order post-TPT | n=4: ratio 0.0045 (P1); n=6: 0.0028–0.0043 (HN3) | 12, 15 | STAGE12_FINDINGS.md, STAGE15_FINDINGS.md |
| ETF geometry over box vertices | d_etf 0.0722 vs d_boxn 0.7122, C=3 L=8 anchor | 12 | STAGE12_FINDINGS.md (P3) |
| Fiber isotropy + dimension (n) scaling | fiber_fraction 0.9934 (n=4) → 0.99963/0.99917 (n=6), tracks isotropic null 1−k/(4^n−1) | 12, 13B, 15 | STAGE12_FINDINGS.md (P4), STAGE13B_FINDINGS.md, STAGE15_FINDINGS.md (HN2) |
| Capacity / TPT transition | TPT fraction 0/5→3/5→5/5 at M/M_c≈0.094→0.188→0.376; TPT-fraction non-decreasing across L∈{3,5,6} (0.3→1.0→1.0) | 9, 14 | RESULTS_V2.md §4, STAGE14_FINDINGS.md (arm 4) |
| Entanglement ceiling | Non-entangling ansatz capped at DLA=12, 0/3 TPT at matched M=96 vs entangling ansatz 5/5 | 10 (Phase 2) | RESULTS_V2.md §5.3 |
| QFI rank structure | Single-input QFI rank saturates at 30 = 2·(dim−1), independent of L∈{8,16,21,26,32} (state-manifold bound, not circuit param bound); batch-aggregated rank ≈ n_params with no M_c saturation below M=384 | 9, 14 | RESULTS_V2.md §4, STAGE14_FINDINGS.md (arm 5) |
| Beta-as-accelerant | beta·mean_margin grows −0.87 → 8.5 (L=8 anchor) post-TPT | 12 | STAGE12_FINDINGS.md (P2) |
| Hardware z-space check (n=4 L=8 R0 anchor, 180 samples, ibm_marrakesh) | NC1_z hardware 0.6749 vs simulator 0.5472; only 17.6% of z-components agree within shot noise (mean abs diff 0.074 vs mean shot-noise std 0.022) — hardware error dominates, sign pattern and order of magnitude preserved. Stage 16b: disagreement is a uniform contraction (lambda=0.7033, R^2=0.9305, HW1 fit-quality met) with angles preserved (HW2 CONFIRMED, sim equiangle_dev_z 0.0969 inside hardware bootstrap 95% CI); a naive additive-covariance NC1_z-prediction formula built on that fit is REFUTED (numerically unstable near Sigma_B's exact null direction). Untrained-parameter hardware control (HW3): NC1_z 94.2x the trained hardware value, CONFIRMED — pipeline tracks real training, not a fixed noise floor. | 16, 16b | STAGE16_FINDINGS.md |

## Refuted hypotheses

| Hypothesis | What killed it |
|---|---|
| Box vertices (class means collapse to CE-optimal box corners, not ETF) | d_boxn (0.71) far exceeds d_etf (0.072) at the L=8 anchor — geometry is ETF-like, not box-like. STAGE12_FINDINGS.md (P3, REFUTED) |
| Expressivity/learnability axis (measurement-family expressivity R0>R1>R2>R3 explains collapse strength) | Apparent full-space weakening across readout families was a circuit-depth confound (R1/R2 had shallower matched-M depths); fixing depth at L=8 collapses the gap to noise (p≥0.42 all arms). STAGE13B_FINDINGS.md (H2, PARTIAL→depth-confound) |
| Beta escape valve (learnable beta's freedom to grow is what prevents within-class collapse) | Freezing beta (arm 6) does not force collapse — within-class variance shrinks *less*, not more (p=0.016, wrong direction); raising the beta ceiling to 500 changes nothing (baseline never approached the original clamp). STAGE14_FINDINGS.md (arms 1, 6, REFUTED) |
| blob_std² inheritance (residual z-space variance floor scales with data spread squared) | Fitted power-law exponent 0.119, over an order of magnitude below the [1.0, 3.0] REFUTED threshold; diagnosed to per-run min-max PCA scaling absorbing blob_std before angle encoding (HR5b CONFIRMED). STAGE14B_FINDINGS.md (HR5, REFUTED) |

## Campaign status: CLOSED, Stage 19 complete (vision datasets + comparative study)

Stage 18 (external-review breadth sprint) is complete: all 7 preregistered
arms scored. B1 (SGD optimizer), B2 (hardware-efficient ansatz), B3 (n=8),
and B5 (second hardware backend, ibm_kingston) are CONFIRMED; B4 (noisy
simulation, inference-only) is PARTIAL (angle-preservation holds at
p=0.001, not at p=0.005); B6 (statistics upgrade) and B7 (mechanism
schematic) are done. Full detail: `STAGE18_FINDINGS.md`, `PREDICTIONS.md`
"Stage 18 preregistration" section, `STATS_APPENDIX.md`.

Stage 19 (ICVGIP 2026 — real vision datasets + comparative study) is
complete: MNIST/Fashion-MNIST replicate the rotation theorem at machine
precision (V1(i) CONFIRMED, 5/5 seeds both datasets); 0/26 VQC runs
reached TPT on real image PCA-4 features at L=8/16/(32-probe) (V1(iii),
factual), while a capacity-matched classical MLP reaches TPT 5/6 — a
genuine ansatz-expressivity gap, not a bug. Comparative study (V2):
classical MLP collapses in feature space (5/6 TPT); VQC no-reupload has
exactly invariant full-space total_var (machine precision) with milder
z-space collapse; VQC reupload collapses z-space >1 order of magnitude
with a small (~2.8%) full-space contraction but noticeably worse test
accuracy; VQC R3 (trainable measurement) is the one REFUTED leg — z-space
NC1 *increased* in 5/6 seeds, opposite the predicted direction. Full
detail, exact numbers, and preregistration deviations (L=8→16 depth bump,
both datasets): `STAGE19_FINDINGS.md`, `PREDICTIONS.md` "Stage 19
preregistration" section. `PLAYBOOK_REMAINING.md` still covers only
Stages 16-17 (historical record; Stages 18-19 are not yet folded in
there).

Stage 20 (real-data TPT attempt + contraction-vs-rotation contrast) is
complete: dropping samples_per_class from 200 (Stage 19's 0/26) to 50/100
closes the real-data TPT gap — 10/10 TPT at spc=50, 9/10 at spc=100, with
NC1_z decay and total_var invariance replicating exactly as on synthetic
data (Arm W1, all three predictions CONFIRMED). A second encoding family
(amplitude embedding, PCA-16, Arm W4) also reaches TPT 4/4 at spc=50 and
independently confirms T5's invariance, though reupload=true there
generalizes at chance level (0.367, C=3) despite reaching TPT — severe
memorization. Arm W2's classical-vs-VQC contrast direction was REFUTED:
classical MLP's absolute within-class scatter tr(Sigma_W) GROWS ~27-163x
despite weak L2 (weight_decay=5e-4, insufficient to bound activation
scale), not the predicted contraction, while VQC total_var stays invariant
at machine precision — arguably a sharper contrast than predicted, reported
factually. Arm W3's full ledger recount (leg-split convention) gives 48
preregistered items (29/7/5/7); under a stricter reading that excludes
pure-deliverable items (B6, B7, V3 — none are directional predictions), the
headline figure is **45 (29 CONFIRMED, 7 REFUTED, 5 PARTIAL, 4
UNSCOREABLE)**, adopted as the reported total — either way this replaces
the paper's stale 35, traced to Stage 19 never being folded in and Stage
18's B6/B7 being dropped. Full detail: `STAGE20_FINDINGS.md`,
`PREDICTIONS.md` "Stage 20 preregistration" section.

## Open items

- Theory notes T1, T2', T3, T5, T6: stated in `THEORY_NOTES.md`, proofs
  owned by the user.
- Paper drafting from `PAPER_SKELETON.md`.

## Pointer map

| Stage | Findings file | SPEC section(s) |
|---|---|---|
| 1–5 (Phase 1) | RESULTS.md (`docs/archive/`) | SPEC.md |
| 7 (reanalysis) | RESULTS_V2.md §3 | SPEC.md, SPEC2_ADDENDUM.md |
| 9 (overparam. transition) | RESULTS_V2.md §4, RESULTS_STAGE9.md | SPEC2_ADDENDUM.md §12 |
| 10 (generalization + entanglement) | RESULTS_V2.md §5 | SPEC2_ADDENDUM.md §13 |
| 12 (measurement-subspace reanalysis) | STAGE12_FINDINGS.md | SPEC3_ADDENDUM.md §17–18 |
| 13 (expressivity sweep, matched-M) | STAGE13B_FINDINGS.md (Task A) | SPEC3_ADDENDUM.md §19 |
| 13B (fixed-depth correction) | STAGE13B_FINDINGS.md (Task B) | SPEC3_ADDENDUM.md §19 |
| 14 (robustness / mechanism arms) | STAGE14_FINDINGS.md | SPEC3_ADDENDUM.md §20 |
| 14B (collapse-by-rotation mechanism) | STAGE14B_FINDINGS.md | SPEC3_ADDENDUM.md §20 |
| 15 (n=6 replication + HR5 diagnosis) | STAGE15_FINDINGS.md | SPEC3_ADDENDUM.md §21 |
| 16 (hardware validation, optional arm) | STAGE16_FINDINGS.md | SPEC3_ADDENDUM.md §22 |
| 17 (remaining) | PLAYBOOK_REMAINING.md | SPEC.md §0, SPEC2_ADDENDUM.md §10, SPEC3_ADDENDUM.md §17 |
| 18 (breadth sprint, external review) | STAGE18_FINDINGS.md | SPEC3_ADDENDUM.md §23 |
| 19 (vision datasets + comparative study, ICVGIP 2026) | STAGE19_FINDINGS.md | PREDICTIONS.md "Stage 19 preregistration" |

Preregistration record for all Stage 12–15 predictions: `PREDICTIONS.md`
(root, untouched by this housekeeping pass).
