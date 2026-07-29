# RESULTS_V3.md — Quantum Neural Collapse: Full Evidence Assembly (Phases 1–3 + Hardware)

Assembled per PLAYBOOK_REMAINING.md Stage 17 / SPEC.md section 0 / SPEC2_ADDENDUM.md
section 10 / SPEC3_ADDENDUM.md section 17. Supersedes RESULTS_V2.md as the
complete record (RESULTS_V2.md's Phase 1/2 content is reproduced here in
condensed form with full traceability preserved; nothing in it is
contradicted). Every number below is copy-pasted from STAGE12–STAGE16_FINDINGS.md,
RESULTS_V2.md, or PREDICTIONS.md's committed thresholds — none is
hand-computed or adjusted here. No new experiments were run to produce this
document (CLAUDE.md invariant 3/4 — this is an assembly pass only).

## Mandatory citations

Papyan, Han, Donoho (2020), PNAS 117(40):24652–24663 (classical NC). Du,
Yang, Tao, Hsieh (2023), PRL 131, 140601 (quantum NC at the static optimum).
San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 (empirical QNC study).
Additional citations used throughout: Larocca, Ju, Garcia-Martin, Coles,
Cerezo (2023), Nat. Comput. Sci. 3, 542–551; Bowles, Ahmed, Schuld (2024),
arXiv:2403.07059; Han, Papyan, Donoho (2022), ICLR; Mixon, Parshall, Pi
(2022, "Neural collapse with unconstrained features", Sampling Theory,
Signal Processing, and Data Analysis 20(2)); Yang, Chen, Li, Xie, Lin, Tao
(NeurIPS 2022, "Inducing Neural Collapse in Imbalanced Learning", fixed-ETF
classifier); Mondal, Chanda, Alawieh, Sukhadiya, Krah, Gonsalves,
Ntolkeras, Rizzoli, Shaib, arXiv:2606.22551 ("QMT" — Quantum Measurement
Temperature). All nine citations' titles/author lists verified against
their arXiv or publisher pages; see `REFERENCES.md` (Stage 17b, Task D).

---

## (a) Narrative arc, one page

**Original question.** Does Neural Collapse (Papyan/Han/Donoho 2020) — the
terminal-phase convergence of within-class features to zero variance and
between-class means to a simplex ETF — occur during the *training* of a
Variational Quantum Circuit (VQC), and if so, does entanglement play a
causal role?

**Reposition (prior work).** Du, Yang, Tao, Hsieh (2023) already proved QNC
holds at the *static global optimum* of a regularized-MSE quantum
classifier. That leaves training dynamics, geometry, and entanglement's
causal role entirely open (SPEC.md section 0.3) — this project's actual
contribution is never "QNC exists" but "how, when, and through what
mechanism a trained VQC approaches or fails to approach it."

**Phase 1 nulls.** A classical MLP control reproduced classical NC cleanly
(instrument validated), but the quantum C=3 anchor never reached the
terminal phase of training (TPT) across 17 tuning attempts at n=6, and the
one quantum anchor that did interpolate (C=2) showed no separation from an
untrained random-init null on unnormalized trace distance (RESULTS_V2.md
§2). Two candidate artifacts: (i) undertrained capacity, (ii) an
unnormalized metric saturated near its concentration-of-measure ceiling.

**Phase 2 capacity transition.** Stage 9's L-sweep (n=4, C=3, blobs3, 40
runs) found TPT reachability jumps sharply with parameter count M
(0/5 → 3/5 → 5/5 at M/M_c_bound ≈ 0.094 → 0.188 → 0.376), consistent with
Larocca et al.'s overparametrization theory — but even fully-interpolating
runs left every *full-space* collapse witness (qnc1_fisher, purity,
equiangle_dev) statistically indistinguishable from the untrained null
(RESULTS_V2.md §4, p ≥ 0.75 under/over comparison). Entanglement ablation
showed a real capacity ceiling (DLA = 12 for the non-entangling ansatz, 0/3
TPT even at matched M=96 vs 5/5 entangling) — the second Phase 1/2 finding
that survives reframing.

**Phase 3 measurement-subspace discovery.** The reconciling hypothesis
(SPEC3 section 17): cross-entropy only ever sees the state through the
C-dimensional projection z = (⟨Z_0⟩,...,⟨Z_{C-1}⟩); full-space witnesses sum
variance over all 4^n − 1 traceless-Hermitian directions and are dominated
by the loss-invisible "fiber" directions. Stage 12's preregistered P1–P5
confirmed z-space collapse (NC1_z decays 2–3 orders of magnitude
post-TPT), confirmed the fiber staying near the untrained-null level
(fiber_fraction 0.9934), and refuted the box-vertex geometry model on
magnitude (d_etf 0.072 vs d_boxn 0.712) — refined (Stage 17b, THEORY_NOTES
T2′) to the sharper statement that class means adopt simplex-ETF
*directions* (which box vertices share at C=3) without ever approaching
the box model's fixed saturation *radius*, because a learnable logit scale
absorbs CE's margin pressure instead.

**Rotation mechanism.** Stage 14B's decisive result: without data
re-uploading, the trained circuit is a single global unitary applied to a
fixed pre-image state, so it can only rotate the constellation of class
states rigidly in Hilbert space — total within-class Hilbert–Schmidt
variance is conserved to machine precision (HR2: ~1e-15 relative change,
5/5 seeds) while NC1_z still decays 1–2 orders of magnitude (HR3). Collapse
is not shrinkage; it is variance redistribution out of the measured
subspace and into the fibers ("collapse by rotation").

**n=6 replication.** Stage 15 lifted the same recipe to n=6 qubits: the
machine-precision total-variance invariance (HN1, ~5e-15), the NC1_z decay
(HN3, ratios 0.0028–0.0043, stronger than n=4's already-strong 0.0045), and
the fiber_fraction ceiling strengthening with dimension exactly as
dimension-counting predicts (HN2: 0.99963/0.99917 > n=4's 0.9934) all
replicate.

**Hardware validation.** Stage 16/16b's inference-only IBM hardware arm
confirmed the pipeline measures a real trained-vs-untrained effect (HW3:
94.2x), found hardware disagreement with the exact simulator is well
described by a single scalar contraction with angles preserved (HW2), and
exposed a genuine numerical instability in the NC1_z-via-pinv formula when
Sigma_B^z is exactly rank-deficient (HW1, informing THEORY_NOTES T6's r_z
recommendation).

---

## (b) Preregistration ledger — every prediction, every stage

All verdicts are reproduced verbatim from their findings files; none are
softened or reworded (CLAUDE.md).

| ID | Stage | Prediction (short) | Verdict | Key number | Findings file |
|---|---|---|---|---|---|
| P1 | 12 | NC1_z / r_z decay post-TPT (L=8, C=2 anchors) | **CONFIRMED** | NC1_z final/init 0.0045 (L8), 0.0071 (C=2) | STAGE12_FINDINGS.md |
| P2 | 12 | beta·mean_margin grows post-TPT | **CONFIRMED** | −0.87 → 8.55 (L8) | STAGE12_FINDINGS.md |
| P3 | 12 | box vertices beat simplex ETF (C=3) | **REFUTED** | d_boxn 0.712 vs d_etf 0.072; correction note (Stage 17b): angle alone doesn't discriminate at C=3 (box vertices also have cosine −1/2), radius is the real test — see THEORY_NOTES T2′ | STAGE12_FINDINGS.md |
| P4 | 12 | fiber decomposition dichotomy | **PARTIAL** | fiber_fraction 0.9934 (≥0.9 met); V_fiber not flat vs null | STAGE12_FINDINGS.md |
| P5 | 12 | untrained null shows no z-space collapse | **CONFIRMED** | Mann-Whitney p=0.000666 | STAGE12_FINDINGS.md |
| H1 | 13B | loss-visible subspace concentrates collapse | **PARTIAL** | concentration 1.42–1.68x (<2x threshold); isotropic-level 1.06–1.16 (met) | STAGE13B_FINDINGS.md |
| H2 | 13B | fixing depth L=8 removes R0-vs-R1/R2 full-space gap | **PARTIAL** | full-space p≥0.42 both arms (met); fiber_excess p=0.0079 for R1 (not met) | STAGE13B_FINDINGS.md |
| H3 | 13B | logit_nc1_z invariant across readout families/depths | **CONFIRMED** | all pairs p≥0.31 (matched-M and fixed-depth) | STAGE13B_FINDINGS.md |
| Arm1 | 14 | beta_max=500 changes nothing (ceiling never binding) | **CONFIRMED** (uninformative) | bit-identical to baseline, p=1 | STAGE14_FINDINGS.md |
| Arm2 | 14 | no slow drift over 5000 epochs | **CONFIRMED** | Spearman rho=−0.0716, p=0.479 | STAGE14_FINDINGS.md |
| Arm3 | 14 | label smoothing 0.1 accelerates within-class shrink | **REFUTED** | right direction, p=0.421 (not significant) | STAGE14_FINDINGS.md |
| Arm4 | 14 | TPT fraction non-decreasing L3→L5→L6 | **CONFIRMED** | 0.3 → 1.0 → 1.0 | STAGE14_FINDINGS.md |
| Arm5 | 14 | single-input QFI rank saturates near 255 | **REFUTED** | saturates at 30 = 2(2^4−1) for all L | STAGE14_FINDINGS.md |
| Arm6 | 14 | frozen beta forces within-class collapse | **REFUTED** | wrong direction, p=0.016; E0 delayed 145.2 vs 35.2 (~4.1x) | STAGE14_FINDINGS.md |
| HR1 | 14B | reupload=true total_var declines <10% | **CONFIRMED** | 0.73% decline (ratio 0.9927) | STAGE14B_FINDINGS.md |
| HR2 | 14B | reupload=false total_var exactly constant | **CONFIRMED** | max rel. change ~1e-15, 5/5 seeds | STAGE14B_FINDINGS.md |
| HR3 | 14B | reupload=false NC1_z still decays ≥1 order | **CONFIRMED** | ratio 0.0209, 5/5 TPT | STAGE14B_FINDINGS.md |
| HR4 | 14B | deeper reuploading circuits contract more | **CONFIRMED** | Spearman rho=−0.451, p=0.0035 | STAGE14B_FINDINGS.md |
| HR5 | 14B | variance floor scales blob_std^2 | **REFUTED** | power=0.119 (outside [1.0,3.0]) | STAGE14B_FINDINGS.md |
| HR5b | 15 | epoch-0 constellation spread flat across blob_std | **CONFIRMED** | rel. range total_var 2.79%, trace_w 6.67% (<20%) | STAGE14B/15_FINDINGS.md |
| HR5c | 15 | final trace_w correlates with own epoch-0 trace_w | **PARTIAL** | rho=0.132>0, p=0.529 (not significant) | STAGE14B/15_FINDINGS.md |
| HN1 | 15 | reupload=false total_var constant at n=6 | **CONFIRMED** | max rel. change ~5e-15, 5/5 seeds | STAGE15_FINDINGS.md |
| HN2 | 15 | fiber_fraction at n=6 exceeds n=4's 0.993 | **CONFIRMED** | 0.99963 (true) / 0.99917 (false) | STAGE15_FINDINGS.md |
| HN3 | 15 | NC1_z decays ≥1 order at n=6, both arms | **CONFIRMED** | ratios 0.0028 / 0.0043 | STAGE15_FINDINGS.md |
| HN4 | 15 | TPT fractions (factual, not scored) | — | 8/8 depth-probe cells 3/3 TPT; campaign 5/5 both arms | STAGE15_FINDINGS.md |
| HW1 | 16b | uniform-contraction fit accounts for NC1_z rise | **REFUTED** | lambda=0.7033, R^2=0.9305 met; NC1_z-pred missed by ~1123x | STAGE16_FINDINGS.md |
| HW2 | 16b | z-space angles invariant under contraction | **CONFIRMED** | sim 0.0969 inside hw bootstrap 95% CI [0.0831,0.1152] | STAGE16_FINDINGS.md |
| HW3 | 16b | untrained hardware NC1_z ≥10x trained hardware | **CONFIRMED** | ratio 94.2x | STAGE16_FINDINGS.md |
| HR6 | 17b | per-class purity + pairwise overlap exactly constant, n=4 | **CONFIRMED** | ~1e-15 relative change, both quantities, 5/5 seeds | STAGE14B_FINDINGS.md |
| HN5 | 17b | per-class purity + pairwise overlap exactly constant, n=6 | **CONFIRMED** | ~1e-15 relative change, both quantities, 5/5 seeds | STAGE15_FINDINGS.md |
| B1 | 18 | SGD momentum generality (n=4 L=8 R0, reupload=false) | **CONFIRMED** | TPT reached, collapse phenomenology replicates | STAGE18_FINDINGS.md |
| B2 | 18 | hardware-efficient ansatz generality | **CONFIRMED** | collapse phenomenology replicates | STAGE18_FINDINGS.md |
| B3 | 18 | n=8 replication | **CONFIRMED** | fiber_fraction/NC1_z ratios <=0.1 (max 0.0272) | STAGE18_FINDINGS.md |
| B4 | 18 | noisy simulation preserves angle structure (inference-only) | **PARTIAL** | holds at p=0.001, not at p=0.005 | STAGE18_FINDINGS.md |
| B5 | 18 | second hardware backend (ibm_kingston) replicates HW2 | **CONFIRMED** | same methodology as HW2 | STAGE18_FINDINGS.md |
| B6 | 18 | statistics upgrade (not a directional prediction) | — | done | STATS_APPENDIX.md |
| B7 | 18 | mechanism schematic (not a directional prediction) | — | done | figs/fig18_mechanism_schematic_* |
| V1(i) | 19 | reupload=false total_var invariant, real vision data (MNIST + FMNIST) | **CONFIRMED** | max rel. change 4.5e-15–9.5e-15, 5/5 seeds both datasets | STAGE19_FINDINGS.md |
| V1(ii) | 19 | NC1_z decay >=1 order in TPT-reaching seeds | **UNSCOREABLE** | 0/20 seeds reached TPT — precondition never met | STAGE19_FINDINGS.md |
| V1(iii) | 19 | TPT fractions (factual, not scored) | — | 0/26 VQC runs reached TPT at L=8/16/32-probe | STAGE19_FINDINGS.md |
| V2(a) | 19 | classical MLP collapses feature + full space | **CONFIRMED** (feature space) | NC1 ratio 0.31–0.91, 5/6 TPT; full-space leg not well-posed for fixed input (scoping gap, not a result) | STAGE19_FINDINGS.md |
| V2(b) | 19 | VQC no-reupload: z-space collapse, full-space exactly invariant | **CONFIRMED** | full-space ~1e-14–1e-15; z-space ratio 0.060 mean | STAGE19_FINDINGS.md |
| V2(c) | 19 | VQC reupload: z-space collapse + small full-space contraction | **CONFIRMED** | z-space ratio 0.0046 mean; full-space contraction 2.8% mean | STAGE19_FINDINGS.md |
| V2(d) | 19 | VQC R3: z-space collapse + small full-space contraction, patterning with (b) | **REFUTED** (z-space direction) | NC1_z *increased* (ratio 1.92 mean, 5/6 seeds); full-space contraction 3.4% mean (CONFIRMED) | STAGE19_FINDINGS.md |

**Tally (superseded — see below):** this document's own row-per-header count
gives 27 CONFIRMED, 7 REFUTED, 5 PARTIAL, 5 not-scored (factual/
unscoreable), across 44 preregistered items, but undercounts rows that
carry two independently-verdicted sub-claims (e.g. V2(d): full-space
CONFIRMED + z-space REFUTED are two distinct verdicts in one row above).

**Corrected headline (Stage 20 Arm W3 recount, leg-split convention — see
`STAGE20_FINDINGS.md` for full stage-by-stage arithmetic):** splitting each
independently-verdicted leg gives **48** items (29 CONFIRMED, 7 REFUTED, 5
PARTIAL, 7 UNSCOREABLE/factual). Excluding the three pure-deliverable items
that are not directional predictions (Stage 18's B6, B7; Stage 19's V3 —
each reported separately below as completed-but-unscored, not folded into
any verdict bucket) gives the adopted headline figure of **45 (29
CONFIRMED, 7 REFUTED, 5 PARTIAL, 4 UNSCOREABLE)**. No softening applied
anywhere in this table (CLAUDE.md); the stale 44/one-row-per-header count
above is retained for traceability, not silently replaced.

**Deliverables (completed, not directional predictions, reported
separately per the stricter 45 reading):** B6 (statistics upgrade,
`STATS_APPENDIX.md`), B7 (mechanism schematic,
`figs/fig18_mechanism_schematic_*`), V3 (scaling figure,
`figs/fig18_stage19_scaling.png`).

---

## (c) Results sections mirroring the arc

### Phase 1 — instrument validation and the C=3 null (Stages 1–5)

Classical MLP control: NC1 decay ~1.3–1.4 orders of magnitude, pairwise
cosines → −0.5 (C=3 ETF target) — instrument validated before any quantum
run. Quantum C=3 anchor (n=6, L=4, M=72) never reached TPT in 17 tuning
attempts (best 98.89% train acc) while classical SVC/RF reached 98.9–100%
on identical features. The C=2 diagnostic that did interpolate showed
post-TPT qnc1_trace (0.9458±0.0059, 5 seeds) statistically indistinguishable
from a 10-seed untrained null (0.9468±0.0063). Entanglement ablation:
removing entanglers dropped tpt_reached 3/3→1/3 and forced entropy_mean to
~8.5e-15 (product states). Figures: `figs/fig1_classical_qnc1.png`,
`figs/fig3_qnc1.png`, `figs/fig5_entropy.png`. Full detail: RESULTS.md
(`docs/archive/`), summarized RESULTS_V2.md §2.

### Stage 7 — normalized-witness reanalysis

Recomputing four normalized witnesses (qnc1_fisher, qnc1_ratio, purity_mean,
plus the original qnc1_trace) at the C=2 anchor against the untrained null:
all four p ≥ 0.44 — the Phase 1 null is a property of the configuration, not
an artifact of the unnormalized metric. Figure:
`results/stage7_reanalysis/figures/four_witness_comparison.png`. Run
traceability: RESULTS_V2.md §8 table (trained: `stage2_vqc_c2_diagnostic_s{0..4}`;
null: `stage5_random_init_s{0..9}`).

### Phase 2 — the overparameterization transition (Stage 9, headline)

40/40 runs (n=4, C=3, blobs3, 8 depths × 5 seeds). TPT fraction 0/5 (L2,
M/M_c=0.094) → 3/5 (L4, 0.188) → 5/5 (L8, 0.376 onward). Where TPT is
reached, E0 falls monotonically (222→8.2 epochs). Full-space collapse
witnesses do NOT jump at the transition: underparameterized (L≤12, n=20) vs
overparameterized (L≥21, n=15), qnc1_fisher p=0.7515, equiangle_dev
p=0.9601. Batch-aggregated QFI rank tracks n_params (no saturation below
M=384), later explained by Stage 14 arm 5 (single-input rank saturates at
the state-manifold bound 30, not the parameter-space bound 255). Figure:
`results/sweeps/transition_L_order_parameter.png`. Full table: RESULTS_V2.md
§4 / RESULTS_STAGE9.md. Entanglement×capacity (Stage 10.3): non-entangling
ansatz capped at DLA=12, 0/3 TPT at matched M=96 vs entangling 5/5. Figure:
`results/sweeps/stage10_ablation_20260713-115624/ablation_bounds.png`.
Dataset generality (Stage 10.1) and CE-vs-MSE (Stage 10.2): figures
`results/sweeps/stage10_datasets_20260713-104835/comparison.png`,
`results/sweeps/stage10_loss_20260713-113927/loss_comparison.png`. Run
traceability: RESULTS_V2.md §8 (full manifest listing, 40+12+10+6 runs).

### Phase 3 — measurement-subspace collapse discovery (Stage 12)

P1–P5 (ledger above). Headline figure — the fiber decomposition dichotomy
(P4): `results/transition_L_L8_aggregate/figures/fiber_dip_diagnostic.png`
(V_fiber vs epoch against the untrained-null band). Supporting figures:
`results/transition_L_L8_aggregate/figures/etf_formation.png` (d_etf and
pairwise z-cosines vs epoch, E0 marked), `.../nc1z_decomposition.png`
(tr(Sigma_W^z), tr(Sigma_B^z) vs epoch). Run traceability: L=8 anchor
5 seeds (`transition_L_L8_s{0..4}_20260712-19{4651,4946,5241,5535,5831}`,
reused from Stage 9); C=2 anchor from Phase 1's `stage2_vqc_c2_diagnostic_s{0..4}`;
untrained null from `stage5_random_init_s{0..9}`.

### Measurement-expressivity mechanism (Stage 13/13B)

H1–H3 (ledger above): the apparent R0>R1>R2 full-space weakening in the
original matched-M sweep was substantially a circuit-depth confound
(R1 L=5, R2 L=3 vs R0's L=8); fixing depth at L=8 removes it for all three
full-space witnesses (H2). The fiber_fraction gap is real but explained by
the isotropic dimension-counting null 1−k/(4^n−1) (k=9 R1, k=18 R2), not by
a strong measurement-subspace collapse signal (H1: concentration only
1.4–1.7x, short of the preregistered 2x). logit_nc1_z is invariant
throughout (H3). Figure: `results/sweeps/stage13_expressivity_aggregate/expressivity.png`
(isotropic null overlaid). Run traceability: matched-M sweep
`results/sweeps/stage13_expressivity_20260714-111146/manifest.json`;
fixed-depth sweep `results/sweeps/stage13b_fixed_depth_20260714-143839/manifest.json`.

### Robustness arms (Stage 14)

Arms 1–6 (ledger above). NC1_z decay itself is robust across every arm
(ratios 0.0038–0.0209 wherever tested); the specific causal mechanisms
tested (beta ceiling, label smoothing, frozen beta) do NOT explain why
within-class z-variance stays roughly flat — arm 6 (frozen beta) directly
refutes the "beta escape valve" story, in the wrong direction. Transition
localization (arm 4) tightens the Phase 2 TPT-onset window. Figures:
`results/sweeps/stage14_transition_localization_aggregate/figures/tpt_fraction.png`
(path corrected from STAGE14_FINDINGS.md's stated `..._assembled/figures/...`,
which does not exist on disk — verified during this assembly pass; the
`_aggregate` directory holds the figure, `_assembled` holds only the
manifest/data), `results/stage14_long_horizon_aggregate/figures/trace_w_drift.png`.
Run traceability: `results/sweeps/stage14_transition_localization_assembled/manifest.json`
(30 runs); long-horizon `stage14_long_horizon_s0_20260714-212225`.

### Collapse by rotation (Stage 14B, centerpiece mechanism)

HR1–HR5 + HR5b/c (ledger above). The rotation-invariance argument
(informal statement in STAGE14B_FINDINGS.md, formalized as THEORY_NOTES T5):
without data re-uploading the trained circuit is a fixed-pre-image unitary
conjugation, an exact Hilbert–Schmidt isometry — HR2 confirms this to
machine precision (~1e-15 relative change in total_var over 600 epochs,
5/5 seeds) while HR3 confirms NC1_z still collapses (ratio 0.0209) under
that exact invariance. HR4 shows deeper *reuploading* circuits (where the
isometry argument does NOT apply) contract total_var more with depth
(rho=−0.451, p=0.0035) — consistent with re-uploading being the mechanism
that breaks the rigid-rotation constraint. HR5's blob_std^2 scaling claim is
refuted and diagnosed (HR5b/c): `pca_encode`'s per-run min-max scaling
absorbs blob_std before the angle encoding ever sees it (epoch-0 spread flat
across blob_std, relative range 2.79%/6.67% ≪ 20%), leaving a real-but-weak,
not-yet-significant within-run inheritance signal (HR5c, rho=0.132, p=0.529).
Figure: `results/stage14b_reanalysis/figures/total_variance_rotation.png`
(reupload=true vs reupload=false overlay). Run traceability:
`configs/sweeps/stage14b_reupload_false.yaml` (5 seeds),
`configs/sweeps/stage14b_blob_std.yaml` + 2 remainder sweeps (15 runs,
manifest `results/sweeps/stage14b_blob_std_assembled/manifest.json`).

### n=6 replication (Stage 15)

HN1–HN4 (ledger above). Depth probe (L∈{8,16,32,64}, both reupload arms, 3
seeds/cell): all 8 cells reach TPT in 3/3 seeds; L8 selected as the smallest
fully-interpolating depth for both arms. 5-seed campaign at L8 replicates
the machine-precision total-variance invariance (HN1) and the NC1_z
collapse (HN3) established at n=4 in Stage 14B, and shows the fiber
dichotomy strengthening with n exactly as dimension-counting predicts (HN2).
Figures: `results/stage15_n6_reanalysis/figures/n4_vs_n6_dichotomy.png`
(fiber_fraction vs epoch, n=4 vs n=6 side by side),
`results/stage15_n6_reanalysis/figures/n6_total_variance_rotation.png`. Run
traceability: depth probe `results/sweeps/stage15_n6_depth_probe_20260716-160429/manifest.json`
(24 runs); campaign `results/sweeps/stage15_n6_campaign_20260716-163857/manifest.json`
(10 runs, 6 seeds reused from the depth probe).

### Hardware validation (Stage 16/16b)

See (d) below.

---

## (d) Hardware section

**Setup.** Inference-only (no training, no tomography), trained n=4 L=8 R0
anchor (seed 0, reupload=true, `results/transition_L_L8_s0_20260712-194651/`),
translated to Qiskit and executed on `ibm_marrakesh` (156 qubits). Translation
gate (Task A): Qiskit statevector vs the PennyLane pipeline, max abs
difference < 1e-6 across all 180 samples — the hard gate that had to pass
before any hardware submission. Budget: 212s ≈ 3.5 min of a 10-QPU-min/28-day
allowance used across both stages (Stage 16: 108s; Stage 16b HW3: 104s),
≈6.5 min remaining.

**Base comparison (Stage 16, Task C).** 180 samples, 2048 shots, Z-basis
readout on qubits 0,1,2, binomial shot-noise error bars.

| Metric | Hardware | Simulator (exact) |
|---|---|---|
| NC1_z | 0.6749 | 0.5472 |
| r_z | 1.7981 | 1.5262 |
| d_box | 1.5549 | 1.4732 |
| mean_margin | 0.1619 | 0.2418 |

Only 17.6% of (sample, class) z-values agree with the simulator within ±1
shot-noise std (mean abs diff 0.074 ≈ 3.4x the mean shot-noise std of
0.022) — the hardware does not agree with the exact simulator within shot
noise alone. The licensed claim (SPEC3 §22) is exactly that agreement/
disagreement statement, nothing stronger.

**Contraction fit (HW1).** lambda = 0.7033 (in preregistered range
[0.6, 0.85]), R^2 = 0.9305 (≥0.9 threshold) — both magnitude/fit-quality
sub-predictions MET. The composite NC1_z-prediction sub-prediction
(lambda^2·Sigma_B^sim + Sigma_B^resid, pinv sandwich) is REFUTED
catastrophically (predicted 758.71 vs actual 0.6749, off by ~1123x).
Diagnosis: for any 3-class z-space, Sigma_B^z is exactly rank ≤ 2 by
construction (class-mean deviations sum to zero); the residual's own
Sigma_B^resid is generically full-rank and does not share that null
direction, so pinv's tolerance treats a small-but-nonzero (not
floating-point-noise) eigenvalue as invertible, blowing up the trace. This
is the concrete instance of THEORY_NOTES T6's NC1-pinv instability, and the
direct motivation for the r_z-alongside-NC1_z recommendation.

**Angle survival (HW2).** Hardware bootstrap (1000 resamples, per-sample
Gaussian perturbation at each z-component's binomial shot-noise std):
equiangle_dev_z mean±std = 0.0983±0.0082, 95% CI [0.0831, 0.1152]. The
exact simulator value (0.0969) falls comfortably inside that CI —
CONFIRMED. The z-space class-mean triangle's *angular* geometry survives
whatever is shrinking its magnitude, consistent with HW1's confirmed
lambda/R^2 (a real, well-fit uniform scaling) even though HW1's specific
NC1_z-prediction formula is unstable.

**Trained-vs-untrained control (HW3).** One new hardware job, untrained
seed-0 initial weights (deterministic re-derivation via `torch.manual_seed(0)`,
no `epoch_0000.pt` artifact needed), same 180 circuits, `ibm_marrakesh`,
2048 shots. Untrained hardware NC1_z = 63.59 vs trained hardware NC1_z =
0.6749 — ratio 94.2x, clearing the preregistered 10x threshold by a wide
margin — CONFIRMED. The Stage 16 hardware pipeline measures a real
trained-vs-untrained effect in the anchor circuit, not a fixed
parameter-independent noise floor.

**NC1-pinv instability caution (HW1 diagnosis) and r_z recommendation.**
NC1_z's `pinv(Sigma_B^z)` formula is acutely sensitive to whichever
direction Sigma_B^z happens to be exactly (or near-exactly) rank-deficient
in — a structural property of C=3 z-space (rank ≤ C−1 = 2), not a
hardware-specific artifact, but one that hardware noise's generically
full-rank residual makes numerically dangerous. Recommendation: report the
scale-free scalar ratio r_z = tr(Sigma_W^z)/tr(Sigma_B^z) alongside NC1_z in
any setting where Sigma_B^z may be near-degenerate (small C, few classes,
hardware noise) — r_z has no matrix inversion and is well-behaved wherever
NC1_z is not.

---

## (e) Relation to prior work

**Du, Yang, Tao, Hsieh (2023), PRL 131, 140601.** Reconciled via what the
loss actually reads: their theorem characterizes the global optimizer of a
regularized-MSE risk in full measurement-operator space; our full-space
witnesses never separate from the untrained null (Phase 1/2) because
cross-entropy only ever reads the C-dimensional z-projection (SPEC3 §17).
Once witnesses are restricted to that projection (z-space), collapse
appears cleanly (Stage 12 P1, Stage 14B HR1–HR4) and strengthens with
measurement learnability in the direction their theorem's free-measurement
setting would predict (Stage 13B H1's directional-but-sub-threshold
concentration). Our setting differs from theirs in three ways that this
project studies and theirs does not: training dynamics (not the static
optimum), density-matrix-space trace distance as the primary tool (not
measurement-operator space), and entanglement's causal role via ablation.
A structural point sharpens this reconciliation further (THEORY_NOTES T5
corollary, Stage 17b): under this architecture's fixed encoding
(reupload=false) and fixed measurement, per-class-mean purity AND pairwise
overlap — the direct density-matrix test of Du et al.'s mutual-orthogonality
prediction — are BOTH exactly invariant under training (HR6/HN5, ~1e-15
relative change, n=4 and n=6). Whatever overlap the untrained model starts
with is exactly what it ends with; Du et al.'s orthogonal-means optimum is
training-*unreachable* in this architecture unless already present at
initialization. Reaching it requires breaking the isometry (data
re-uploading, HR1/HR4, or a trainable measurement block, Stage 13's R3).

**Larocca, Ju, Garcia-Martin, Coles, Cerezo (2023), Nat. Comput. Sci.**
Transition location: TPT reachability jumps at M/M_c_bound ≈ 0.19–0.38,
well below their DLA upper bound (4^n−1=255 at n=4), consistent with a
theory that only bounds M_c from above. QFI rank structure: the
batch-aggregated Gram matrix (averaged over 16 inputs) tracks n_params with
no saturation below M=384 (>255), which does NOT match their single-orbit
DLA-rank-saturation criterion — resolved by Stage 14 arm 5: single-input QFI
rank saturates at exactly 30 for every depth tested, which is
2·(2^4−1) = 2·(dim−1), the real dimension of CP^15 (the pure-state manifold
bound for a 4-qubit Hilbert space), not the circuit's raw parameter-space
bound 4^n−1=255 that the batch-aggregated rank approaches. A single input
pins the model to one trajectory through state space; no amount of circuit
depth can push a single-input QFI rank past the state manifold's own
dimension. This state-manifold bound (2(2^n−1)) is a structural fact
independent of ansatz choice, distinct from — and tighter than, for
single-input measurements — the DLA bound their theory uses.

**Bowles, Ahmed, Schuld (2024), arXiv:2403.07059.** Entanglement tension
scoped strictly to interpolation, not generalization: our non-entangling
ansatz reached TPT (exact interpolation) in 0/3 seeds at both tested depths
while the entangling ansatz reached 5/5 at matched M=96 (RESULTS_V2.md §5.3)
— in tension with their "entanglement is often dispensable" finding only on
the axis of exact training-set interpolation. Bowles et al.'s claim concerns
test-set generalization performance; we measure 100% train accuracy. The two
findings can coexist and likely do — the DLA capacity ceiling (12 for the
non-entangling ansatz at any n=4 depth) supplies the mechanism for our
result (no amount of depth raises a capped DLA), which says nothing about
whether an entangling model that DOES interpolate generalizes better or
worse than a non-entangling one that doesn't.

**San Sebastian, Canizo, Orus (2026), arXiv:2602.08485.** Complementary
axis: they work in measurement-operator space across losses and observable
sets; we work in density-matrix space (trace distance, HS ratios, purity)
plus z-space (SPEC3 §18) and study training dynamics, entanglement
ablation, and the capacity transition, none of which their study addresses.
Our CE-vs-MSE result (RESULTS_V2.md §5.2, witnesses indistinguishable at
matched budget except overlap_offdiag, p=0.0079) is consistent with their
across-loss NC tendency, with the budget-matching caveat that our MSE arm
never reached TPT.

**Han, Papyan, Donoho (2022), ICLR.** Classical NC dynamics under MSE loss
supplied the precedent for our label-smoothing arm (Stage 14 arm 3, testing
whether the classical NC2-acceleration effect of label smoothing is
detectable in the quantum z-space setting) — REFUTED at n=5 seeds (right
direction, not significant, p=0.421), reported factually as a
sample-size-limited null, not a contradiction of their classical result.

**QMT paper, arXiv:2606.22551 (Mondal et al.).** Introduces Quantum
Measurement Temperature (QMT), a learnable pre-loss rescaling of measurement
outputs that "increases gradient magnitude and variance, thereby improving
loss sensitivity" — conceptually the same role our learnable `beta` plays
(logit inverse-temperature as an accelerant of loss-visible signal). Their
paper does not discuss a frozen-temperature ablation or TPT-onset delay; our
own frozen-beta arm (Stage 14 arm 6) supplies an independent empirical echo
of the same "temperature helps training reach the loss-sensitive regime
faster" mechanism: freezing beta at a fixed value delays TPT onset by
~4.1x (mean E0 145.2 vs 35.2 epochs) without forcing additional within-class
collapse (arm 6 REFUTED as an escape-valve mechanism, but E0 delay itself is
consistent with the QMT framing of temperature as an accelerant rather than
a cause of collapse per se).

---

## (f) Limitations

1. **Two n values only.** n=4 (primary) and n=6 (Stage 15 replication).
   Both z-space-collapse/fiber-no-collapse dichotomy and its dimension-count
   strengthening are observed at exactly two points — a trend fit, not
   established scaling law.
2. **blobs3 synthetic primary.** The headline sweep and the Stage 14B/15
   rotation-mechanism arms use a controlled synthetic Gaussian-blob dataset
   by design (isolating landscape/mechanism effects from data pathology).
   Real-data external validity rests on the smaller (3-seed) Stage 10
   dataset arms, of which mnist_pca4 is the only non-synthetic one, run only
   at Phase 2 (full-space witnesses), not re-run through the Phase 3
   z-space/fiber pipeline.
3. **Single entropy bipartition.** `entropy_mean` (Phase 1/2) uses one fixed
   cut (`first_half`); entanglement structure across other bipartitions is
   unmeasured throughout.
4. **HR5's pipeline-normalization caveat.** The blob_std^2 variance-floor
   claim's refutation (HR5) is diagnosed, not merely observed: `pca_encode`'s
   per-run min-max scaling to [-pi, pi] absorbs blob_std's raw spread before
   angle encoding, so the refutation is specific to this encoding pipeline,
   not necessarily to the underlying physical claim under a different
   (non-normalizing) encoding — untested here.
5. **Hardware = one backend, one trained seed, inference only.** All
   hardware evidence (Stage 16/16b) is `ibm_marrakesh`, the n=4 L=8 R0
   anchor's seed-0 trained parameters (plus one untrained control), 180
   samples, no training on hardware, no tomography, no repetition across
   backends or seeds. HW1's contraction-fit lambda/R^2 and HW2's angle
   survival are single-anchor, single-backend findings.
6. **The DLA bound is an upper bound, not M_c.** 4^n−1 bounds M_c from above
   throughout (Phase 2 and the state-manifold-bound discussion in (e)); no
   exact M_c is reported anywhere in this project.
7. **Carried over from Phase 1/2 (RESULTS_V2.md §7).** The Phase 1
   trained-vs-null comparison is at a C=2 anchor where equiangle_dev is
   degenerate; the MSE arm's non-interpolation means the Du-theorem
   comparison remains budget-matched dynamics only, not optimum-vs-optimum.

---

**Full run_id traceability** for every figure in this document is given
inline in section (c) and in RESULTS_V2.md §8 for all Phase 1/2 figures;
Stage 12–16b run_ids are given in their respective findings files
(STAGE12_FINDINGS.md through STAGE16_FINDINGS.md) and are not duplicated
here beyond the manifest paths already cited above.
