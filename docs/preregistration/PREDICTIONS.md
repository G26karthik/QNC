# PREDICTIONS.md -- Stage 12 preregistration

Preregistered per PLAYBOOK3.md Stage 12 / SPEC3_ADDENDUM.md sections 17-18,
written and committed BEFORE any section 18 metric is computed. Tests the
measurement-subspace collapse (MSC) hypothesis: CE depends on the state only
through z = (<Z_0>,...,<Z_{C-1}>), so collapse is predicted in z-space and
NOT in the loss-invisible fiber directions that dominate the full-space
witnesses studied in Phase 1/2.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS -- classical
NC definition. Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 -- quantum NC at
the optimum (unconstrained-features setting). San Sebastian, Canizo, Orus
(2026), arXiv:2602.08485 -- empirical QNC study across observable sets. QNC is
not "unexplored": Du et al. proved the static optimum; Phase 3 studies where
in Hilbert space training-time collapse pressure actually lands.

## Arms under test

- **L=8 anchor**: `transition_L_L8_s0`..`s4` (Phase 2, n=4, C=3, blobs3,
  5/5 seeds reached TPT per Stage 9).
- **C=2 anchor**: `stage2_vqc_c2_diagnostic_s0`..`s4` (Phase 1, n=6, C=2).
- **Untrained null**: `stage5_random_init_s0`..`s9` (10 seeds, epoch 0 only,
  no training).

## Predictions

### P1 -- NC1_z and r_z decay post-TPT (L=8 anchor, C=2 anchor)

`nc1_z` and `r_z` (SPEC3 section 18.1) computed at every checkpointed epoch.
Predicted: both DECAY by >= 1 order of magnitude (10x) from their pre-TPT
(early epoch) value to their final post-TPT value, for both anchors.
Scoring: CONFIRMED if final/initial <= 0.1 for both metrics at both anchors;
PARTIAL if true for one anchor or one metric only; REFUTED otherwise.

### P2 -- beta * mean_margin grows post-TPT (L=8 anchor, C=2 anchor)

`mean_margin` (SPEC3 section 18.2) and logged `beta` combine into the
effective CE margin beta * mean_margin. Predicted: monotonically GROWS
(net increase, final > initial) post-TPT at both anchors, tracking the
already-logged TPT crossing (e0_epoch / tpt_reached fields).
Scoring: CONFIRMED if beta*mean_margin(final) > beta*mean_margin(pre-TPT)
at both anchors; PARTIAL if true at one; REFUTED otherwise.

### P3 -- box vertices beat ETF at C=3 (L=8 anchor only, C=3 required)

`d_box` (raw) and `equiangle_dev_z` (d_etf, SPEC3 section 18.3) computed at
final epoch. Predicted: class means zbar_c approach BOX VERTICES more
closely than a simplex ETF, i.e. normalized d_boxn = d_box/2 ends smaller
than d_etf = equiangle_dev_z at the final logged epoch, averaged across the
5 seeds.
Scoring: CONFIRMED if mean(d_boxn) < mean(d_etf) at final epoch across
seeds; REFUTED if the reverse; PARTIAL if within noise (overlapping
mean +/- std bands).

### P4 -- fiber decomposition dichotomy (L=8 anchor, decisive witness)

SPEC3 section 18.4: V_meas, V_fiber, fiber_fraction computed at every
checkpointed epoch, using the orthonormalized Pauli-Z readout projector
(verified by the <Z_j,Z_k>_HS = delta_jk * 2^n unit test, section 18.5).
Predicted:
- V_meas DECAYS post-TPT (tracks NC1_z direction).
- V_fiber STAYS at null-baseline level (no directional trend outside the
  untrained-null's own seed-to-seed spread; not a decay, not a comparable
  growth).
- fiber_fraction -> close to 1 by the final epoch (predicted >= 0.9).
Scoring: CONFIRMED if all three sub-predictions hold at the 5-seed mean;
PARTIAL if 2 of 3 hold; REFUTED if <= 1 holds.

### P5 -- untrained null shows no z-space collapse (sanity arm)

`nc1_z` computed at epoch 0 for the 10-seed random-init null. Predicted:
NC1_z does NOT decay relative to itself (single-timepoint arm -- this
predicts the null's NC1_z distribution is NOT collapsed, i.e. comparable in
scale to the L=8 anchor's own PRE-TPT (early-epoch) NC1_z, and clearly
distinguishable from the anchor's POST-TPT (final-epoch) NC1_z).
Scoring test: Mann-Whitney U, null NC1_z (n=10) vs L=8-anchor final-epoch
NC1_z (n=5). CONFIRMED if p < 0.05 (null statistically distinct from
collapsed trained state); REFUTED if p >= 0.05.

## Interpretation gate (PLAYBOOK3.md Stage 12)

Four outcomes, scored strictly against the above before any Stage 13
decision:
1. P1/P4 confirmed, fiber flat -> MSC hypothesis confirmed, proceed to
   Stage 13 unmodified.
2. P1/P4 confirmed + P3 confirmed (box beats ETF) -> add geometry claim to
   title.
3. P1 refuted (NC1_z does not decay) -> redirect Stage 13; do not launch it
   unmodified.
4. Mixed across seeds/anchors -> scope claims by regime.

No prediction here is softened after the fact (CLAUDE.md: report numbers,
never claim collapse/no-collapse in comments without them).

## Stage 13B preregistration (H1-H3) -- depth-confound correction

Written and committed BEFORE any Stage 13B metric is computed. Motivated by
two post-hoc observations on the existing Stage 13 matched-M sweep
(`results/sweeps/stage13_expressivity_20260714-111146/`): (i) the matched-M
padding scheme gave R1/R2 much shallower circuits than R0/R3 (R0 L=8, R1 L=5,
R2 L=3, R3 L=6 -- a depth confound, not a pure measurement-expressivity
effect); (ii) R1/R2's observed fiber_fraction (0.9666, 0.9285) sits almost
exactly on the isotropic dimension-counting null 1 - k/(4^n-1) for their
respective measured-span dimension k (0.9647 for k=9, 0.9294 for k=18),
while R0/R3 (k=3, null 0.9882) sit measurably above it (0.9934, 0.9931).

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

### Arms under test

- **Matched-M (existing, Task A, checkpoints only)**: R0/R1/R2/R3 final
  checkpoints from `results/sweeps/stage13_expressivity_20260714-111146/manifest.json`.
- **Fixed-depth (new, Task B)**: R1 and R2 rerun with circuit fixed at L=8
  (96 circuit params), heads unchanged (30 and 57 params respectively,
  reported separately, not padded), 5 seeds each, blobs3 C=3 n=4, 600 epochs.
  R0/R3 unchanged (already L=8/L=6, not rerun).

### H1 -- loss-visible subspace concentrates collapse within the measured span

Within the k-dim measured span of R1/R2 (SPEC3 section 18.4's measured
subspace, span of that arm's orthonormalized Pauli basis), decompose
within-class deviation variance V_meas into the C-dim image of the trained
head's row space (Gram-Schmidt-orthonormalized W rows mapped into operator
space via the arm's basis) and its (k-C)-dim in-span complement. Per-dimension
variance: v_wrow/C (loss-visible) vs v_complement/(k-C) (loss-invisible but
still inside the measured span). Isotropic per-dimension expectation across
the whole span: V_meas/k.
Predicted: v_wrow/C is enriched at least 2x above the isotropic per-dim rate
V_meas/k (collapse concentrates where the loss can see); v_complement/(k-C)
sits within +/-20% of V_meas/k (no concentration, no depletion -- isotropic
level), for both R1 and R2 at the final checkpoint (5-seed mean).
Scoring: CONFIRMED if both sub-predictions hold for both arms; PARTIAL if
they hold for one arm only, or for one sub-prediction across both arms;
REFUTED otherwise. Computed on BOTH the matched-M checkpoints (Task A) and
the fixed-depth checkpoints (Task B, once available) as a depth-robustness
check; the matched-M result is the primary scoring basis, the fixed-depth
result is reported alongside as robustness evidence (a mismatch between the
two is itself reported, not averaged away).

### H2 -- fixing depth at L=8 removes the R0-vs-R1/R2 full-space gap

With circuit depth fixed at L=8 for R1 and R2 (Task B), full-space witnesses
(qnc1_fisher, purity_mean, overlap_offdiag) and fiber-excess-above-null
(fiber_fraction minus that arm's isotropic null 1 - k/(4^n-1)) are predicted
to match R0 within noise -- i.e. the matched-M sweep's apparent R0>R1>R2
weakening was a depth artifact, not a measurement-expressivity effect.
Scoring: CONFIRMED if Mann-Whitney U (R0 vs fixed-depth R1, R0 vs fixed-depth
R2, n=5 each) gives p >= 0.05 on all three full-space witnesses AND the
fiber-excess-above-null mean +/- std bands overlap R0's for both arms;
PARTIAL if this holds for one arm, or for a strict subset of witnesses;
REFUTED if p < 0.05 on a majority of witnesses for both arms or excess bands
do not overlap.

### H3 -- NC1_z on logits is invariant across arms

`logit_nc1_z`: NC1_z (SPEC3 section 18.1 formula) computed directly on each
arm's actual model output (the C-dim vector fed to the CE loss -- beta*z for
R0/R3, W f for R1/R2), NOT the pre-head feature-space `feature_nc1_z` already
logged. Predicted: invariant across R0/R1/R2/R3, both in the matched-M set
(Task A) and the fixed-depth set (Task B for R1/R2).
Scoring: CONFIRMED if Mann-Whitney U (R0 vs each other arm, n=5 each) gives
p >= 0.05 for all pairs in both the matched-M and fixed-depth comparisons;
PARTIAL if this holds in one of the two comparisons, or for a subset of
arm pairs; REFUTED if p < 0.05 for a majority of pairs in both comparisons.

## Interpretation gate (Stage 13B)

H1/H2/H3 scored CONFIRMED/REFUTED/PARTIAL with numbers in
STAGE13B_FINDINGS.md; the original Stage 13 matched-M table and figure are
kept as data but flagged as depth-confounded in interpretation, not deleted
or altered. No softening after the fact.

## Stage 14 preregistration -- mechanism tests of the scaling-escape hypothesis

Written and committed BEFORE any Stage 14 metric is computed. Binding
reframing (per this stage's task prompt): Stage 14 arms are no longer generic
"robustness of the dichotomy" checks -- they are MECHANISM TESTS of the
scaling-escape hypothesis: NC1_z decay is driven by between-class separation
(tr Sigma_B^z growth) while within-class variance (tr Sigma_W^z) barely
shrinks; learnable beta is hypothesized to be the escape valve that lets
training avoid variability collapse (full within-class collapse to a point).

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

### Baseline

**L=8 R0 baseline**: the existing Stage 13 matched-M sweep's R0 arm
(`results/sweeps/stage13_expressivity_20260714-111146/`, label R0, 5 seeds,
L=8, learnable beta, no label smoothing). Not rerun. Final-epoch values:
`nc1_z` 0.2795 +/- 0.06583, and `trace_w`/`trace_b` recomputed fresh in
Stage 14 Task B alongside the new arms for an apples-to-apples Mann-Whitney U
(rather than reusing STAGE12_FINDINGS.md's L=8-anchor numbers, which come
from a different config, `stage9_transition.yaml`, not `stage13_expressivity.yaml`).

### Witness set (arms 1, 2, 3, 6 -- all R0 readout)

Every arm in this group logs, at the final checkpointed epoch (5-seed mean
+/- std, or the single seed for arm 2): `trace_w` = tr(Sigma_W^z), `trace_b`
= tr(Sigma_B^z), `nc1_z`, and `concentration_factor`.

**concentration_factor is preregistered as a pipeline sanity check, not a
mechanism test, for these four arms.** R0's readout has no trainable linear
head (fixed Z_c operators only) -- the measured subspace (dimension C=3) IS
the loss-visible subspace in its entirety, so the H1 decomposition (SPEC3
Stage 13B: v_wrow-image vs in-span complement) is degenerate: complement_dim
= 0, wrow_dim = meas_dim = C. Therefore concentration_factor =
meas_per_dim / wrow_per_dim = 1.0 EXACTLY, by construction, for every R0 arm.
Prediction: CONFIRMED iff the computed value is 1.0 within 1e-9 absolute
tolerance for all seeds of all four arms; any deviation indicates a bug in
the recompute pipeline, not a scientific finding.

### Arm 1 -- beta_max = 500 (5 seeds)

Config: L=8 R0 baseline with `model.beta_max: 500` (vs baseline's 50).
Prediction: within-class shrink is weaker than or equal to baseline
(`trace_w` final for arm1 >= `trace_w` final for baseline, allowing for
seed-to-seed noise -- i.e. NOT a significant decrease, Mann-Whitney U
p >= 0.05 or arm1 mean >= baseline mean); `nc1_z` still decays by >= 1 order
of magnitude from its own epoch-0 value (separation-driven collapse persists
even with a much larger beta ceiling).
Scoring: CONFIRMED if both sub-predictions hold; PARTIAL if one holds;
REFUTED if neither holds.

### Arm 2 -- long horizon, 5000 epochs (1 seed, seed 0)

Config: L=8 R0 baseline, `train.epochs: 5000`,
`logging.dense_until: 5000, logging.log_interval_dense: 50` (checkpoint and
log every 50 epochs throughout, ~100 checkpoints total).
Prediction: Spearman correlation of `trace_w` vs log(epoch), computed over
all checkpointed epochs strictly after this run's own `e0_epoch` (TPT onset),
shows NO significant downward drift -- i.e. NOT (rho < 0 AND p < 0.05). This
converts the Stage 12/13 null (`trace_w` does not shrink over 600 epochs)
into "does not shrink over 5000 epochs either," ruling out slow-drift
collapse that a short run would miss.
Scoring: CONFIRMED if p >= 0.05, or p < 0.05 with rho >= 0 (drift, if any, is
not a shrink); REFUTED if rho < 0 and p < 0.05 (significant downward drift).

### Arm 3 -- label smoothing 0.1 (5 seeds)

Config: L=8 R0 baseline with `train.label_smoothing: 0.1` (new
`nn.CrossEntropyLoss(label_smoothing=0.1)` support added to train.py Task B).
Prediction: within-class shrink is STRONGER than baseline -- `trace_w` final
for arm3 is lower than baseline's `trace_w` final (Mann-Whitney U p < 0.05,
arm3 mean < baseline mean). Label smoothing classically accelerates NC2/NC1
in the classical literature (Papyan/Han/Donoho 2020 discuss temperature/
smoothing sensitivity); this tests whether ANY collapse pressure reaches
within-class z-space variance in the quantum setting.
Scoring: CONFIRMED if the Mann-Whitney U condition holds; REFUTED if arm3
mean >= baseline mean or p >= 0.05; no PARTIAL (single directional claim).

### Arm 4 -- transition localization, L in {3,5,6} (10 seeds each, unchanged from SPEC3 section 20)

Config: `stage9_transition.yaml` base, `model.layers` in {3,5,6}, seeds 0-9
each (30 runs total). No z-space recompute -- uses only the already-live-
logged `tpt_reached`/`e0_epoch` fields. Prediction (original SPEC3 section 20
prediction, unchanged): TPT-fraction (seeds reaching TPT / 10) at L=3 is
lower than at L=5 and L=6, tightening the Phase 2 "TPT reached somewhere
between 9% and 38% of M_c" statement into a monotonic-in-L curve over these
three closely-spaced depths.
Scoring: CONFIRMED if TPT-fraction is non-decreasing L3 -> L5 -> L6;
PARTIAL if non-monotonic but L6 fraction > L3 fraction; REFUTED if
L6 fraction <= L3 fraction.

### Arm 5 -- single-input QFI rank fix, L in {8,16,21,26,32} (original SPEC3 section 20 prediction, unchanged)

No new training. Epoch-0 seeded reconstruction (seed 0) off the existing
Stage 9 `transition_L` configs. QFI rank computed for 4 separate single
training inputs (batch size 1 each, `qfi_matrix`/`qfi_rank` already support
arbitrary batch size), reporting each of the 4 ranks and their mean, compared
to the already-logged batch-16 epoch-0 `qfi_rank` from the original Phase 2
Stage 9 sweep (`results/transition_L_L*_s0/metrics.jsonl`, epoch 0).
Prediction: single-input ranks saturate near the full su(16) dimension bound
(255) for L >= 21, unlike the batch-aggregated rank, which did not saturate
in Phase 2 -- closing the methodological loose end that batch aggregation
under-reports per-state QFI rank.
Scoring: CONFIRMED if mean single-input rank at L=26 and L=32 is
>= 0.95 * 255 (242); PARTIAL if true at only one of {L26, L32}; REFUTED if
neither.

### Arm 6 -- frozen beta, beta=5.0 non-learnable (5 seeds, NEW)

Config: L=8 R0 baseline with `model.beta_learnable: false`,
`model.beta0: 5.0` (matches baseline's initial beta value, but held fixed
throughout training -- tests whether beta's freedom to grow, not just its
initial value, is the escape valve).
Prediction: within-class shrink is STRONGER than baseline (`trace_w` final
for arm6 lower than baseline's, Mann-Whitney U p < 0.05, arm6 mean < baseline
mean) -- without a learnable beta to grow the effective margin
(beta * mean_margin), the model is hypothesized to fall back on shrinking
within-class z-space variance to fit the loss. E0 (TPT onset) may occur
later than baseline, or TPT may not be reached at all within 600 epochs;
report whichever occurs factually, this is not itself scored.
Scoring: CONFIRMED if the Mann-Whitney U condition holds; REFUTED if arm6
mean >= baseline mean or p >= 0.05; no PARTIAL (single directional claim).

## Interpretation gate (Stage 14)

Six arms scored CONFIRMED/REFUTED/PARTIAL independently in
STAGE14_FINDINGS.md, each against the specific threshold above. No softening
after the fact (CLAUDE.md). If arm1 or arm3 or arm6 REFUTES the scaling-
escape hypothesis (i.e. `trace_w` does not move in the predicted direction),
that is reported as a direct constraint on the mechanism story, not
downplayed relative to the arms that confirm.

## Stage 14B preregistration -- collapse by rotation

Written and committed BEFORE any Stage 14B metric is computed. Binding
reframing (per this stage's task prompt, not derived from SPEC3_ADDENDUM.md):
"collapse by rotation." The variational circuit is unitary; for angle
encoding WITHOUT data re-uploading (`model.reupload: false`), the total
within-class Hilbert-Schmidt variance of the state constellation --
`total_var = V_meas + V_fiber` (SPEC3 section 18.4, summed via Parseval's
identity over the full 4^n - 1 traceless-Hermitian directions) -- is
mathematically invariant under training: the encoding fixes each sample's
state rho_0(x) once, and the trained unitary U(theta) acts as
rho(x) = U(theta) rho_0(x) U(theta)^dagger, an isometry of Hilbert-Schmidt
distance. Apparent collapse (NC1_z decay, Stage 12/13/14) must then come
entirely from ROTATING the rigid constellation so within-class spread hides
in loss-invisible fiber directions (V_fiber growing at V_meas's expense,
total_var unchanged), not from any real shrinkage of the constellation.
Re-uploading (`model.reupload: true`, all prior stages) breaks exact
rigidity -- the encoding gate is reapplied at each layer under trained
weights, so the state map is no longer a single global unitary applied to a
fixed pre-image -- and is hypothesized to permit limited real contraction.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

### HR1 -- reupload=true total variance declines only slightly (free, existing checkpoints)

At the L=8 anchor (`transition_L_L8_s0..s4`, reupload=true, Stage 9
manifest), `total_var` = `v_meas` + `v_fiber` per checkpointed epoch
(recomputed via the existing `reanalyze_l8_anchor_zspace` per-epoch records,
no new training). Predicted: the trajectory declines by < 10% total
(final/initial >= 0.9), and that small decline is the ENTIRE source of the
Stage 12 V_fiber dip below the untrained-null level (i.e. the dip is a small
real contraction riding on an otherwise near-rigid constellation, not a
large one).
Scoring: CONFIRMED if 5-seed-mean final/initial `total_var` ratio >= 0.9;
PARTIAL if in [0.8, 0.9); REFUTED if < 0.8 (a decline that large would mean
"only rotation" is itself wrong even for the re-uploading case).

### HR2 -- reupload=false: total variance is exactly constant (new runs, 5 seeds)

Config: `stage14_mechanism_base.yaml` with `model.reupload: false` (otherwise
identical L=8 anchor: blobs3, C=3, n=4, blob_std=0.8, 600 epochs), 5 seeds.
Predicted: `total_var` is constant across training to numerical precision --
relative change `abs(total_var[t] - total_var[0]) / total_var[0] < 1e-6` at
EVERY checkpointed epoch, for every seed. Predicted consequence: the
V_fiber-below-null dip (Stage 12's decisive witness) VANISHES in this arm
(V_fiber cannot rise above its epoch-0 value if total_var is exactly fixed
and V_meas can only draw from that same fixed budget).
Scoring: CONFIRMED if the < 1e-6 relative-change bound holds at every
checkpoint for all 5 seeds; PARTIAL if it holds for a majority of
seed-checkpoint pairs but not all; REFUTED if a majority of seeds show
relative change >= 1e-6 at any checkpoint (this is the rigidity theorem's
direct numerical fingerprint -- REFUTED here would falsify the exact-unitary
argument, not just weaken it, so no softer PARTIAL band below the majority
threshold).

### HR3 -- reupload=false runs still show NC1_z decay if they reach TPT

Predicted: among whichever reupload=false seeds reach TPT (`tpt_reached`),
`nc1_z` still decays by >= 1 order of magnitude (10x) from its own epoch-0
value to its final value -- rotation alone suffices to produce the z-space
collapse diagnostics, since NC1_z depends only on the measured-subspace
projection, which the rigid rotation can still concentrate. TPT fraction
among the 5 seeds is reported factually regardless of outcome; per this
stage's task prompt, a TPT fraction of 0/5 REFUTES NOTHING in HR2 (total-
variance invariance is checkable on non-TPT runs from epoch 0 to their final
epoch regardless of whether CE reached the TPT threshold) and is itself
informative for HR3 (rotation alone may not suffice to reach TPT at C=3
within 600 epochs).
Scoring: computed only among TPT-reaching seeds (if any). CONFIRMED if
NC1_z(final)/NC1_z(epoch0) <= 0.1 for the TPT-reaching-seed mean; PARTIAL if
true for some but not all TPT-reaching seeds; REFUTED if false for the
TPT-reaching-seed mean. If 0/5 seeds reach TPT, HR3 is reported as
"untestable, 0/5 TPT" -- not scored CONFIRMED/REFUTED/PARTIAL, and this
outcome does not count against HR2.

### HR4 -- deeper reuploading circuits contract more (free, existing checkpoints)

Across the Stage 9 L sweep (`transition_L_manifest.json`, L in
{2,4,8,12,16,21,26,32}, 5 seeds each, all reupload=true), final `trace_w`
(z-space within-class variance, SPEC3 section 18.1) recomputed via
`reanalyze_stage9_sweep_zspace_final` (already exists, no new training).
Predicted: `trace_w` (final epoch) DECREASES with L -- Spearman rho < 0
across the 40 (L, seed) pairs, p < 0.05.
Scoring: CONFIRMED if Spearman rho < 0 and p < 0.05; PARTIAL if rho < 0 but
p >= 0.05; REFUTED if rho >= 0.

### HR5 -- variance floor scales with blob_std^2 (new runs, 15 total)

Config: `stage14_mechanism_base.yaml` (L=8 anchor, reupload=true) with
`data.blob_std` in {0.2, 0.4, 0.8}, 5 seeds each (15 runs). Final `trace_w`
(z-space within-class variance) fitted as a power law
`trace_w_final ~ blob_std^power` (log-log linear regression across the 3
blob_std values' 5-seed means; report fitted `power` and `R^2`).
Predicted: `power` in [1.5, 2.5] -- the variance floor is inherited from the
data through a near-rigid map (an exactly rigid isometry with fixed relative
geometry would give power = 2 exactly; deviations from 2 bound how far
reupload=true departs from rigidity).
Scoring: CONFIRMED if fitted power in [1.5, 2.5] and R^2 >= 0.8; PARTIAL if
power in [1.5, 2.5] but R^2 < 0.8, or power just outside [1.0, 3.0]; REFUTED
if power outside [1.0, 3.0] or R^2 < 0.5.

## Interpretation gate (Stage 14B)

HR1/HR4 (free) and HR2/HR3/HR5 (20 new runs) scored CONFIRMED/REFUTED/
PARTIAL independently in STAGE14B_FINDINGS.md against the thresholds above.
No softening after the fact (CLAUDE.md). HR2 is the decisive arm: if it
CONFIRMS (total_var invariant to < 1e-6 relative change under
reupload=false), that is strong numerical evidence for the rotation
mechanism and directly explains why Stage 12/13/14's full-space witnesses
never showed real collapse under reupload=true either (rotation dominates,
with only the small HR1-bounded real contraction on top). If HR2 REFUTES,
the exact-unitary argument is wrong as stated and must be revisited before
any formal write-up in THEORY_NOTES (Stage 17) -- report the numbers
factually either way.

## Stage 15 preregistration -- HR5 diagnosis (checkpoints only) + n=6 replication (HN1-HN4)

Written and committed BEFORE any Stage 15 metric is computed. Binding
context (per this stage's task prompt): the rotation mechanism is confirmed
(HR1-HR4, HR2 at machine precision, STAGE14B_FINDINGS.md). HR5 (variance
floor scales with blob_std^2) was REFUTED. Suspected diagnosis: `pca_encode`'s
min-max scaling to [-pi, pi] (SPEC.md section 2) is fit per-run on that run's
own train split, so a run's blob_std sets the RAW spread of its data but the
scaler then normalizes that spread away before encoding -- the encoded
constellation on the Bloch-sphere-like manifold never inherits blob_std's
scale. HR5b/HR5c test this diagnosis directly; both are checkpoints-only, no
new training.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

### HR5b -- epoch-0 constellation spread is flat across blob_std (checkpoints only)

For the 15 blob_std runs (`configs/sweeps/stage14b_blob_std.yaml` +
remainders, blob_std in {0.2, 0.4, 0.8}, 5 seeds each), epoch-0 seeded
reconstruction (no training): `total_var` (SPEC3 section 18.4,
`total_within_class_hs_variance`) and `trace_w` (SPEC3 section 18.1) per
blob_std arm (5-seed mean each). Predicted: BOTH metrics' relative range
across the three arm means, `(max - min) / mean(all 3 means)`, is < 20% --
the epoch-0 (pre-training, post-min-max-scaling) constellation spread is
nearly flat across a 4x change in raw blob_std, attributing HR5's refutation
to the min-max normalization absorbing blob_std before encoding.
Scoring: CONFIRMED if relative range < 20% for both `total_var` and
`trace_w`; PARTIAL if true for one of the two; REFUTED if true for neither.

### HR5c -- corrected inheritance test: final trace_w correlates with the SAME run's own epoch-0 trace_w (checkpoints only)

Pooled across three run sets that never had their epoch-0 trace_w compared
to their own final trace_w before now: the 15 blob_std runs, the 5
`stage14b_reupload_false` runs, and the 5 `transition_L_L8` (L=8 anchor)
seeds -- n=25 (run, epoch0_trace_w, final_trace_w) pairs total. Spearman
correlation of final `trace_w` vs that same run's epoch-0 `trace_w` (all 25
pairs pooled, one correlation, not per-arm). Predicted: rho > 0, p < 0.05 --
whatever set a run's initial z-space within-class variance (dataset
realization, seed, init) has some persistent effect on where it lands after
training, even though the aggregate level is a near-data-independent floor
(HR5's refutation was about the arm-mean LEVEL not tracking blob_std^2, not
about run-to-run correlation within a fixed setup).
Scoring: CONFIRMED if rho > 0 and p < 0.05; PARTIAL if rho > 0 but p >= 0.05;
REFUTED if rho <= 0.

### Interpretation gate (HR5 diagnosis)

HR5b and HR5c scored independently in STAGE14B_FINDINGS.md's new "HR5
diagnosis" section. No softening after the fact (CLAUDE.md). Both PARTIAL or
REFUTED does not reopen HR5 (already REFUTED, reported factually); it would
instead mean the specific min-max-absorption diagnosis needs revision, which
is itself reported.

---

Scale replication per SPEC3_ADDENDUM.md section 21 / PLAYBOOK3.md Stage 15,
upgraded with the confirmed rotation-mechanism arms (reupload=true AND
reupload=false, HR1-HR4): blobs3 lifted to R^6 (same center-distance 3.0 /
blob_std 0.8 recipe, 60 samples/class), n=6 qubits, C=3. Depth probe L in
{8, 16, 32, 64}, 3 seeds per depth per reupload arm (24 runs); then 5 seeds
at the smallest fully-interpolating depth per arm (3/3 probe seeds reaching
TPT) with the full SPEC2 + SPEC3 section 18 witness set, `total_var` logged
at every checkpointed epoch. n=4 fiber_fraction reference value: ~0.993 (L=8
anchor, STAGE12_FINDINGS.md P4).

### HN1 -- reupload=false total_var constant at n=6 (machine precision)

Predicted: identical numerical fingerprint to HR2 at n=4 -- relative change
`abs(total_var[t] - total_var[0]) / total_var[0] < 1e-6` at EVERY
checkpointed epoch, for all 5 campaign seeds of the n=6 reupload=false arm.
Scoring: CONFIRMED if the bound holds at every checkpoint for all 5 seeds;
PARTIAL if it holds for a majority of seed-checkpoint pairs but not all;
REFUTED if a majority of seeds show relative change >= 1e-6 at any
checkpoint (mirrors HR2's no-soft-PARTIAL-below-majority rule).

### HN2 -- fiber_fraction at n=6 exceeds the n=4 value in all TPT-reaching arms

Predicted: final-epoch `fiber_fraction` (5-seed mean) > 0.993 for every
TPT-reaching n=6 campaign arm (reupload=true and/or reupload=false, whichever
reach TPT) -- dimension counting (3 measured of 4095 fiber directions at n=6,
vs 3 of 255 at n=4) predicts the dichotomy strengthens with n.
Scoring: CONFIRMED if true for every TPT-reaching arm; PARTIAL if true for
some but not all; REFUTED if true for none. If 0 arms reach TPT, reported as
"untestable, 0 TPT-reaching arms," not scored.

### HN3 -- NC1_z decays >= 1 order of magnitude in TPT-reaching arms, both reupload settings

Predicted: among whichever n=6 campaign seeds reach TPT, `nc1_z(final) /
nc1_z(epoch0) <= 0.1` (5-seed mean among TPT-reaching seeds), separately for
the reupload=true and reupload=false arms.
Scoring: computed only among TPT-reaching seeds per arm. CONFIRMED if the
ratio bound holds for the TPT-reaching-seed mean in BOTH arms; PARTIAL if
true for one arm only; REFUTED if false for both. An arm with 0/5 TPT is
reported as "untestable, 0/5 TPT" for that arm and does not count against
the other arm's scoring.

### HN4 -- TPT fractions reported factually (no threshold)

Depth-probe TPT fraction (seeds reaching TPT / 3) per (L, reupload-arm) cell,
and campaign TPT fraction (/5) at the selected depth per arm, reported as a
factual table in STAGE15_FINDINGS.md. Not scored CONFIRMED/REFUTED/PARTIAL.

### Interpretation gate (Stage 15)

HR5b/HR5c and HN1-HN4 scored independently in STAGE15_FINDINGS.md against
the thresholds above. No softening after the fact (CLAUDE.md). Wall time
reported after the first n=6 run completes; per this stage's task prompt, if
any single run exceeds ~2 hours the user is consulted before continuing
rather than silently reducing scope (fewer seeds, fewer depths, etc.).

---

## Stage 16b preregistration -- contraction decomposition + untrained hardware control

Written and committed BEFORE any Stage 16b metric is computed. Context
(STAGE16_FINDINGS.md, Stage 16): the trained n=4 L=8 R0 anchor's hardware
z-values (`ibm_marrakesh`, 180 samples, 2048 shots) disagreed with the exact
simulator beyond shot noise alone (17.6% of (sample,component) pairs within
1 shot-noise std), with a pattern consistent with a uniform contraction of
all z-components toward the origin (NC1_z hardware 0.6749 vs simulator
0.5472). Stage 16b tests that contraction hypothesis quantitatively and adds
an untrained-parameter hardware control.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

### HW1 -- uniform-contraction fit accounts for the NC1_z rise (existing Stage 16 counts, no new hardware)

Least-squares fit `z_hw = lambda * z_sim + residual`, no intercept (pure
scaling, matching the contraction picture), pooled across all 540
(sample, component) pairs (180 samples x 3 z-components) into one scalar
`lambda`. `R^2 = 1 - SS_res/SS_tot` (residual sum of squares vs total sum
of squares around the observed hardware mean, conventional definition).
Decomposition check: using the actual empirical residual `eps = z_hw -
lambda*z_sim` (not an isotropic-noise assumption), form
`Sigma_W^resid`/`Sigma_B^resid` (SPEC3 section 18.1 formulas, applied to
`eps`) and predict `Sigma_W^z_hw_pred = lambda^2 * Sigma_W^z_sim +
Sigma_W^resid`, `Sigma_B^z_hw_pred = lambda^2 * Sigma_B^z_sim +
Sigma_B^resid`, then `nc1_z_pred = (1/C) tr(Sigma_W^z_hw_pred @
pinv(Sigma_B^z_hw_pred))`.
Predicted: `lambda` in [0.6, 0.85], `R^2 >= 0.9`, and `nc1_z_pred` within
15% of the observed hardware `nc1_z` (0.6749): `abs(nc1_z_pred - 0.6749) /
0.6749 <= 0.15`.
Scoring: CONFIRMED if all three sub-predictions hold; PARTIAL if the
lambda/R^2 fit-quality conditions hold but the NC1_z-prediction condition
misses by up to 2x the 15% band (i.e. <= 30% off), or vice versa; REFUTED
otherwise.

### HW2 -- z-space angles are invariant under the contraction (existing Stage 16 counts, no new hardware)

`equiangle_dev_z` (SPEC3 section 18.3, pairwise cosines of centered z-space
class means vs the ETF target -1/(C-1)) computed for hardware and for the
exact simulator on the same 180 samples. Hardware error bars via bootstrap:
>= 1000 resamples, each resample redrawing every sample's z_i by adding
independent Gaussian noise per component with std equal to that sample's
already-computed binomial shot-noise std (`z_from_counts`'s `z_err`),
recomputing `equiangle_dev_z` on the perturbed set each time, giving a
bootstrap distribution (report mean, std, 95% CI).
Predicted: the simulator's exact (noiseless) `equiangle_dev_z` point value
falls inside the hardware bootstrap 95% CI -- i.e. the angular geometry of
the class-mean triangle is preserved under the contraction (magnitude
shrinks, angles do not), consistent with HW1's pure-scaling picture.
Scoring: CONFIRMED if the simulator value falls inside the hardware 95% CI;
PARTIAL if outside the 95% but inside the 99.7% (3-sigma) CI; REFUTED if
outside even that.

### HW3 -- untrained-parameter hardware control (ONE new hardware job)

Same 180 circuits (same data, same backend family, same 2048 shots, same
Sampler-batch submission), but with the anchor's seed-0 UNTRAINED
parameters instead of the trained `epoch_0600.pt` checkpoint. No
`epoch_0000.pt` file was saved by the original training run, but this
repo's established epoch-0 reconstruction convention (Stage 5's
`stage5_random_init` arms; HR5b's "epoch-0 seeded reconstruction, no
training") applies directly here: `train.py` calls `torch.manual_seed(seed)`
immediately before constructing the `VQC`, and nothing between that call and
construction (data loading, PCA) touches the global torch RNG (both use
independently seeded generators) -- so `torch.manual_seed(0)` followed by
constructing the same `VQC(n_qubits=4, num_classes=3, n_layers=8,
reupload=True, entangling=True, encoding="angle", readout_family="R0")`
deterministically reproduces the exact untrained initial weights the real
training run started from.
Predicted: untrained hardware `nc1_z` exceeds the trained hardware value
(0.6749) by at least one order of magnitude (10x) -- an untrained circuit's
z-space class means are expected to be near the coordinate origin (no
learned separation), so `Sigma_B^z` is small and `NC1_z` (which divides by
it) is correspondingly large; this also serves as a sanity check that the
Stage 16 hardware pipeline is measuring a real trained-vs-untrained
difference, not a fixed hardware-noise artifact independent of the circuit.
Scoring: CONFIRMED if `nc1_z_untrained / 0.6749 >= 10`; PARTIAL if in
[3, 10); REFUTED if < 3.

### Interpretation gate (Stage 16b)

HW1/HW2/HW3 scored independently in a new "Stage 16b" section of
STAGE16_FINDINGS.md against the thresholds above. No softening after the
fact (CLAUDE.md). The licensed claim stays exactly SPEC3 section 22's
(agreement or disagreement within shot noise plus hardware error); HW1/HW2
characterize the disagreement's structure (a contraction, angle-preserving)
rather than upgrading the claim, and HW3 is a control, not a new collapse
claim.

---

## Stage 17b preregistration -- exact invariants under reupload=false (HR6/HN5)

Written and committed BEFORE any Stage 17b metric is computed. Context
(STAGE14B_FINDINGS.md's rotation-invariance argument, formalized as
THEORY_NOTES.md T5): for angle encoding without data re-uploading, the
trained circuit is a single global unitary U(theta) applied to a fixed
pre-image state, rho(x;theta) = U(theta) rho_0(x) U(theta)^dagger.
HR2/HN1 already confirmed this for `total_var` (the Parseval-summed
within-class HS variance) to machine precision (~1e-15 at n=4, ~5e-15 at
n=6). The isometry argument is stronger than total_var alone: unitary
conjugation preserves ANY Hilbert-Schmidt-computable quantity of the full
constellation that does not depend on a choice of basis broken by theta,
including per-class-mean purity and pairwise class-mean overlap (both are
polynomial in the rho_bar_c's, which transform covariantly under the same
global U(theta)). No new training; reuses the exact 5 reupload_false
checkpoint sets already loaded for HR2 (n=4, `STAGE14B_REUPLOAD_FALSE_RUN_IDS`)
and HN1 (n=6, `stage15_n6_campaign`'s `L8_reupload_false` cell, 5 seeds).

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

### HR6 -- per-class purity and pairwise overlap constant at n=4 (existing HR2 checkpoints, no new training)

For each of the 5 reupload=false runs (`STAGE14B_REUPLOAD_FALSE_RUN_IDS`),
recompute at every checkpointed epoch: `mean_class_purity` =
(1/C) sum_c Tr(rho_bar_c^2) and `class_mean_overlap` =
mean_{c<c'} Tr(rho_bar_c @ rho_bar_c') (both already-defined metrics
functions `purity` and `overlap_offdiag`, applied to the per-class-mean
density matrices `means` already computed for the fiber decomposition --
no new formula). Report, per seed, the maximum relative change from the
epoch-0 value across all checkpoints:
`max_t abs(v[t] - v[0]) / abs(v[0])` for both quantities.

Predicted: max relative change < 1e-6 at every checkpoint, for both
quantities, in all 5 seeds (machine precision, mirroring HR2's threshold
and result).

Scoring: CONFIRMED if the < 1e-6 bound holds for both quantities in 5/5
seeds; PARTIAL if it holds for one quantity in 5/5 seeds but not the other,
or holds for both in at least 3/5 seeds; REFUTED otherwise.

### HN5 -- per-class purity and pairwise overlap constant at n=6 (existing HN1 checkpoints, no new training)

Identical procedure and threshold to HR6, applied to the 5 seeds of
`stage15_n6_campaign`'s `L8_reupload_false` cell (the same checkpoint set
already used for HN1).

Predicted: max relative change < 1e-6 at every checkpoint, for both
quantities, in all 5 seeds.

Scoring: identical to HR6 (CONFIRMED / PARTIAL / REFUTED conditions).

### Interpretation gate (Stage 17b, Task A)

HR6/HN5 scored independently against the thresholds above, appended to
STAGE14B_FINDINGS.md (HR6) and STAGE15_FINDINGS.md (HN5) and the RESULTS_V3
ledger. No softening after the fact (CLAUDE.md). These are free
verifications (existing checkpoints only) of a stronger form of the same
rotation-invariance argument HR2/HN1 already confirmed for total_var; a
REFUTED or PARTIAL result here would indicate the isometry argument does
not extend to non-additive (product-of-two-states) quantities the way T5
as currently stated assumes, and would require narrowing T5's scope before
the paper draft, not softening HR2/HN1's own machine-precision result.

---

## Stage 18 preregistration -- breadth sprint (external review response)

Written and committed BEFORE any Stage 18 metric is computed, including the
two timing/feasibility probes (B3, B4) explicitly authorized to run before
this document's commit lands in the same session -- the probes measure
wall-clock time and basic sanity output only; the scored quantities below
(fiber_fraction, lambda_sim(p), equiangle_dev_z, etc.) are not read as
"the answer" until after this file is committed. All decision/trim rules
below are fixed now, precisely so that applying them after seeing probe
output is not post-hoc goalpost-moving.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

Anchor unless stated: blobs3, n=4, L=8, R0 (as used throughout Stages 12-16b).

### B1 -- optimizer generality: SGD momentum=0.9 (n=4 L=8 anchor, reupload=false)

Procedure: lr pick (0.05 vs 0.1, 1 seed each, whichever reaches TPT or
lower final loss wins) at reupload=false (the exact-invariance config used
throughout HR2/HR6/HN1/HN5, chosen here specifically because it is the
configuration with a crisp, falsifiable machine-precision threshold --
reupload=true's total_var is only approximately conserved, per HR1's 0.73%
decline, so it would not cleanly test "the invariance theorem is
optimizer-independent"). Then 5 seeds at the chosen lr, full witness set
incl. total_var.

Predicted: TPT reached in >=4/5 seeds; total_var machine-precision
invariance holds under SGD-momentum exactly as it does under Adam (max
relative change < 1e-6 at every checkpoint, mirroring HR2's threshold);
NC1_z decays >=1 order of magnitude (ratio <= 0.1) in TPT-reaching seeds.

Scoring: CONFIRMED if TPT-fraction and both thresholds (total_var < 1e-6,
NC1_z ratio <= 0.1) hold in >=4/5 seeds; PARTIAL if TPT-fraction holds but
one threshold fails, or both hold in exactly 3/5; REFUTED otherwise.

### B2 -- ansatz generality: hardware-efficient ansatz (per-qubit RY-RZ + ring CZ), n=4 anchor

Procedure (completing the structure proposed in-session): depth probe on
reupload=true only (this establishes interpolation feasibility -- the
question being probed), depths {8, 16, 32, 64} x 3 seeds, same probe
methodology as Stage 15's `stage15_n6_depth_probe`. At the smallest
depth reaching TPT in 3/3 probe seeds, run the full campaign as both arms
at 5 seeds each: 5x reupload=true, 5x reupload=false (mirroring Stage 15's
`stage15_n6_campaign` structure exactly). The reupload=false arm at full
seed count is mandatory, not optional -- it is this arm's headline claim.

Predicted:
- Machine-precision total_var invariance for the 5 reupload=false seeds
  (max relative change < 1e-6 at every checkpoint, 5/5 seeds) -- the
  theorem (T5) is a property of the encoding/no-reupload structure, not of
  the entangling-layer ansatz family, so it should replicate exactly under
  hardware-efficient entanglers too.
- NC1_z decays >=1 order of magnitude (ratio <= 0.1) in TPT-reaching seeds
  of the reupload=true 5-seed set.

Scoring: CONFIRMED if both bullets hold in 5/5 seeds; PARTIAL if the
total_var bullet holds in 5/5 but NC1_z holds in 3-4/5 (or vice versa);
REFUTED if either holds in <3/5.

### B3 -- third qubit count: n=8 (blobs3 lifted to R^8)

**Timing probe (authorized to run before this commit lands):** one seed,
L=8, reupload=true, full witness set, using the same "lift" recipe as
Stage 15's n=6 base config (blob_dist=3.0, blob_std=0.8, 60 samples/class,
600 epochs) with `ambient_dim`/`n_qubits` = 8. Records `elapsed_seconds`
only, same format as `stage15_n6_timing_probe.json`.

**Trim rule (fixed now, mechanical, no post-hoc judgment call permitted):**
extrapolate the full plan's unattended wall time as
`probe_seconds x 3 seeds x len(depths)` for the depth-probe portion plus
`probe_seconds x 5` for the final campaign, using the probe's L=8 timing
as a stand-in for L=8 and doubling it per depth-doubling (matching the
~1.8x-per-doubling pattern observed in the n=6 depth-probe timings) to
estimate L=16/L=32. If that extrapolated total exceeds ~12 hours:
- Drop the middle depth. Depth probe becomes {8, 32} x 3 seeds (6 runs,
  down from 9) instead of {8, 16, 32} x 3 seeds.
- The final 5-seed campaign still runs at the smallest depth (of the
  trimmed set) that reaches TPT in 3/3 probe seeds.
- If neither 8 nor 32 reaches 3/3 TPT in the trimmed probe, report the
  depth-probe result as-is (TPT fractions, factual, not scored) and do not
  extrapolate further without checking back in.
If the extrapolated total is <= ~12 hours, run the untrimmed {8, 16, 32}
plan as specified in the original arm description.

Predicted: fiber_fraction at n=8 exceeds the n=6 value (0.99963, HN2),
consistent with the isotropic-null prediction 1 - k/(4^n - 1) evaluated at
n=8 (k = number of measured components = C = 3, giving null
1 - 3/65535 = 0.9999542); reupload=false total_var invariance replicates
to machine precision (max relative change < 1e-6, matching HR2/HN1/HR6/HN5).

Scoring: CONFIRMED if fiber_fraction > 0.99963 AND the invariance bound
holds in >=4/5 (or however many seeds the trim rule leaves) campaign seeds;
PARTIAL if only one of the two holds; REFUTED if neither holds, or if
fiber_fraction is not even monotonically higher than the n=6 value while
n=4 < n=6 was (i.e. the scaling trend itself breaks).

### B4 -- noisy simulation, restructured as inference-only (apples-to-apples with the hardware protocol)

Restructured in-session from the original training-based design (3 seeds
x 2 noise levels, retrained from scratch) to an inference-only design that
mirrors Stage 16/16b exactly: apply a per-gate depolarizing channel
(`default.mixed`, exact/shots=null density-matrix simulation, no sampling
noise) to the SAME already-trained anchor used by the hardware arm --
`results/transition_L_L8_s0_20260712-194651/checkpoints/epoch_0600.pt`
(n=4, L=8, R0, reupload=true, seed 0) -- and measure z over the same 180
blobs3 samples (144 train + 36 test) Stage 16 used. No training, no new
optimizer runs; "seeds" collapse to none needed because this is a
deterministic exact-noise computation, not a stochastic one -- p in
{0.001, 0.005}, one deterministic result each.

**Feasibility probe (authorized to run before this commit lands):** one
p value (0.001), time the 180-sample inference pass end-to-end.

Predicted: measured z-values are uniformly contracted relative to the
noiseless (`default.qubit`) simulation at both p values; fitting the same
contraction-fit methodology as HW1 (`lambda_sim(p)`, least-squares
z_noisy = lambda * z_exact) gives lambda_sim(0.001) > lambda_sim(0.005)
(monotonically decreasing in p, both < 1); class-mean angular geometry is
preserved: relative change in `equiangle_dev_z` between noiseless and each
noisy condition is < 10% (tighter than the hardware arm's shot-noise-driven
bootstrap-CI threshold, since this is an exact deterministic computation
with no sampling error to average over) -- this is the simulator-side
bridge to HW1/HW2's hardware contraction result.

Scoring: CONFIRMED if lambda_sim is monotonic decreasing in p AND both
p-values' equiangle_dev_z relative change is < 10%; PARTIAL if monotonicity
holds but one equiangle_dev_z change is in [10%, 25%); REFUTED otherwise.

**Stretch goal (not required for Stage 18 completion, scored separately
only if run):** if the feasibility probe shows `default.mixed` per-epoch
cost keeps a full ~600-epoch training run under ~30 minutes, add a noisy
TRAINING arm (3 seeds x 2 noise levels, from-scratch training under the
noise model) as originally specified. Not a precondition for closing
Stage 18; its absence is not a REFUTED result for B4.

### B5 -- hardware, second backend: ibm_kingston (ONE new hardware job)

Same protocol as Stage 16/16b: trained n=4 L=8 R0 seed-0 anchor
(`epoch_0600.pt`), same 180 samples, same Sampler-batch submission,
>=2048 shots/circuit if budget allows. Standing gates (unchanged from
Stage 16): present a cost estimate and get explicit in-session go-ahead
immediately before submission; abort if the estimate extrapolates past 3
minutes; submit as one batched job. Sanity-check the estimate against the
two prior batch jobs on this project (`ibm_marrakesh` full-batch, 104s;
Stage 16b HW3 control, ~104s) before submitting.

Budget accounting (for the record, not a prediction): 10 QPU-min / 28-day
account total; ~3.5 min used by Stage 16+16b; ~6.5 min remaining before
this job. A ~104s (~1.7 min) job leaves ~4.8 min in reserve afterward --
the last planned hardware spend for this project; the remainder stays
unspent.

Predicted: contraction-fit lambda_kingston in [0.55, 0.85] (same range as
Stage 16b's HW1 fit on ibm_marrakesh, lambda=0.7033); angle survival --
the noiseless simulator's equiangle_dev_z (0.0969) falls inside the
ibm_kingston bootstrap 95% CI, same methodology as HW2.

Scoring: CONFIRMED if both hold; PARTIAL if lambda is outside [0.55, 0.85]
but the angle-survival bullet holds (or vice versa); REFUTED if neither
holds, or if the job cannot be completed within the budget gate (reported
factually as an incomplete arm, not scored as REFUTED).

### B6 -- statistics upgrade (not a scored prediction; methodology deliverable)

For every existing ledger comparison (RESULTS_V3.md (b), all 30 items) and
every new Stage 18 comparison (B1-B5 above), add Cliff's delta effect size
and bootstrap 95% CIs; apply Holm-Bonferroni correction across the full
family of confirmatory (CONFIRMED-eligible) tests and report which
conclusions survive correction. Written to `STATS_APPENDIX.md`. Not
scored CONFIRMED/REFUTED -- a methodology addition, reported factually
(which prior conclusions survive Holm-Bonferroni, which do not).

### B7 -- conceptual mechanism schematic (not a scored prediction; deliverable)

Four-panel figure: encoded constellation -> global rotation -> measured
projection -> apparent collapse. Publication quality, sized for both
single-column and double-column (journal) layout. Saved under the
project's figures output convention. Not scored -- a deliverable.

### Interpretation gate (Stage 18)

B1-B5 scored independently against the thresholds above in
`STAGE18_FINDINGS.md`, exactly as every prior stage's arms have been. No
softening after the fact (CLAUDE.md). B6/B7 are factual/methodology
deliverables, not predictions, and are reported as such rather than
force-fit into CONFIRMED/REFUTED. The B3 and B4 trim/restructuring rules
above are fixed at commit time; any deviation from them after seeing probe
output must be reported as a deviation, not silently applied.

---

## Stage 19 preregistration -- vision datasets + comparative study (ICVGIP 2026)

Written and committed BEFORE any Stage 19 metric is computed, including the
Fashion-MNIST TPT-reachability probe explicitly authorized to run before
this document's commit lands in the same session (per the same convention
as Stage 18's B3/B4 probes) -- the probe measures TPT-fraction only; the
scored quantities below are not read as "the answer" until after this file
is committed.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

Anchor unless stated: n=4, L=8, R0, blobs3-recipe hyperparameters carried
over from Stages 12-16b/18 wherever not otherwise specified.

### Arm V1 -- real vision datasets, full witness set

Datasets: MNIST classes {0,1,2} (existing `dataset: mnist` loader,
verified working -- no separate "mnist_pca4" function exists in this
codebase, `dataset: mnist` + PCA-to-n_qubits inside `train_vqc` already
does the job); Fashion-MNIST classes {0,1,2} = T-shirt/top, Trouser,
Pullover (new `load_fashion_mnist_subset`, same IDX-gz recipe, different
mirror). PCA-4, 200 samples/class, test_fraction as anchor config.

**Feasibility probe (authorized to run before this commit lands):**
Fashion-MNIST only, L=8, reupload=true, 2 seeds, full witness set.
**Trim rule (fixed now):** if fewer than 2/2 probe seeds reach TPT, fall
back to L=16 for the Fashion-MNIST campaign only (MNIST stays at L=8); if
2/2 reach TPT, run the untrimmed L=8 plan. Any deviation from this rule
after seeing probe output is reported as a deviation, not silently applied.

Campaign: 2 datasets x 2 reupload arms (true/false) x 5 seeds = 20 runs.

Predicted:
- (i) reupload=false total_var is invariant to machine precision (max
  relative change < 1e-6 at every checkpoint) on BOTH real datasets --
  the rotation theorem (T5) is dataset-independent, a property of the
  no-reupload encoding structure, not of blobs3 specifically.
- (ii) NC1_z decays >=1 order of magnitude (ratio <= 0.1) in TPT-reaching
  seeds, both datasets, both reupload arms.
- (iii) TPT fractions reported factually (not scored), per dataset/arm.

Scoring: CONFIRMED per dataset if (i) holds in 5/5 seeds AND (ii) holds in
TPT-reaching seeds with ratio <= 0.1 in the median; PARTIAL if (i) holds in
5/5 but (ii)'s median ratio is in (0.1, 0.3]; REFUTED if (i) fails in >=1/5
seeds or (ii)'s median ratio exceeds 0.3.

### Arm V2 -- comparative study on identical PCA-4 features

Approaches, 3 seeds each, on MNIST-PCA4 and Fashion-MNIST-PCA4 (6 runs per
approach, 24 total):
- (a) Classical MLP, capacity-matched to the VQC (NOT the Stage 1 control's
  784-512-512-64 architecture, which would be absurdly overparameterized
  for a 4-d input and collapse trivially): 4 -> 64 -> 16 -> C, ReLU, SGD
  momentum 0.9, weight_decay 5e-4. Penultimate width 16 = 2^n_qubits (n=4)
  matches the VQC's full statevector dimension -- a principled capacity
  match, not an arbitrary shrink. Classical NC1/NC2 on the 16-d penultimate
  features via existing `metrics.py` functions.
- (b) VQC, fixed angle encoding, no re-upload (n=4, L=8, R0).
- (c) VQC, data re-uploading (n=4, L=8, R0).
- (d) VQC, trainable measurement, readout_family R3 at L=8 (measurement_layers=2).
  This is a NEW config, not a reuse of Stage 13B's fixed-depth correction --
  Stage 13B only reran R1/R2 at L=8; its R3 arm was reused unchanged from
  the matched-M sweep at L=6. Using that L=6 config here would reintroduce
  the exact depth confound Stage 13B was built to diagnose, so a fresh
  L=8 R3 config is built instead and reported as such.

Comparison table columns: test accuracy, feature/z-space NC1 final ratio
(final/init), full-space collapse classification (invariant / mildly
contracting / collapsing, by total_var behavior across training), within-
class contraction factor (trace_w final/init).

Predicted (directional, per approach, both datasets):
- (a) collapses in BOTH feature space and full input space (classical MLP
  has no rotation-invariance constraint -- nothing analogous to T5 protects
  full-space variance).
- (b) z-space collapse only; full-space total_var exactly invariant
  (machine precision, mirroring HR2/HN1/HR6/HN5).
- (c) z-space collapse plus a small, non-zero full-space contraction
  (mirroring HR1's 0.73% decline at n=4 -- re-uploading breaks the
  fixed-pre-image isometry argument).
- (d) z-space collapse plus small full-space contraction, similar order to
  (c) -- R3's extra measurement-layer parameters sit downstream of the
  same no-reupload state-preparation unitary as (b), so the same rotation
  argument (T5) should still apply to the state before measurement;
  predicted to pattern with (b) (full-space invariant), not (c), unless
  the measurement layers themselves are shown to inject reupload-like
  structure (not expected by construction, but not yet checked).

Scoring per approach (aggregated across both datasets, 6 seeds): CONFIRMED
if the total_var behavior classification (invariant/contracting/collapsing)
matches the directional prediction in >=5/6 seeds AND z-space NC1 ratio
<=0.1 in TPT-reaching seeds; PARTIAL if the total_var classification holds
in 4/6 or the NC1 threshold is met only on one dataset; REFUTED otherwise.

### Arm V3 -- scaling figure (free, existing data, no new runs)

fiber_fraction vs qubit count (n=4, 6, 8) using already-reported values
(n=4: 0.9934, Stage 12 P4; n=6: 0.99963/0.99917 true/false, Stage 15 HN2;
n=8: Stage 18 B3's reported value), with the isotropic-null prediction
curve 1 - 3/(4^n - 1) overlaid (k=3=C, matching the k used in all prior
isotropic-null comparisons). Publication quality, column-sized. Not a new
prediction -- a visualization of already-scored results (HN2, B3), so not
independently CONFIRMED/REFUTED here.

### Interpretation gate (Stage 19)

V1 and V2 scored independently against the thresholds above in
`STAGE19_FINDINGS.md`, exactly as every prior stage's arms have been. No
softening after the fact (CLAUDE.md). V3 is a figure deliverable built
from already-scored numbers, reported factually. The V1 Fashion-MNIST
trim rule above is fixed at commit time; any deviation after seeing probe
output is reported as a deviation, not silently applied.

## Stage 20 preregistration -- real-data TPT attempt + contraction-vs-rotation contrast

Written and committed BEFORE any Stage 20 metric is computed. Arm W1's
probe (4 runs) is explicitly authorized to run before this document's
commit lands in the same session, per the identical convention used for
Stage 18's B3/B4 probes and Stage 19's Fashion-MNIST probe -- the probe
measures TPT-fraction only, and its role (gating the full campaign vs. a
reduced fallback) is fixed here, not chosen after seeing its output.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

Anchor unless stated: MNIST classes {0,1,2}, PCA-4, n=4, R0, L=16 (Stage 19's
established working depth for this dataset/qubit count -- L=8 never reached
TPT per Stage 19, so is not retried here).

### Arm W1 -- real-data TPT attempt (the gap Stage 19 left open)

**Probe (authorized to run before this commit lands):** samples_per_class=50,
reupload in {true, false}, 2 seeds each (seeds 0-1) = 4 runs. spc=50 is the
cell most likely to interpolate (least data, easiest to overfit toward a
training-accuracy=1 terminal point); if it fails, spc=100 cannot succeed
either under the same TPT criterion, so the probe is fully informative
about whether the full campaign is worth running -- same probe-then-
campaign discipline as Stages 15, 18, 19.

**Trim rule (fixed now):** if >=1/4 probe seeds reach TPT, run the full
campaign: 2 samples_per_class values {50,100} x 2 reupload arms x 5 seeds
(0-4) = 20 runs (the probe's 4 runs count toward the 5-seed spc=50 cells,
3 additional seeds needed there; spc=100 cells run fresh, 5 seeds each).
If 0/4 probe seeds reach TPT, drop to 3 seeds/cell (seeds 0-2) x 2 spc x 2
reupload = 12 runs total, and report the 0/N TPT fraction factually as a
result, not a failure. Any deviation from this rule after seeing probe
output is reported as a deviation, not silently applied.

Predicted:
- (i) TPT fraction increases as samples_per_class decreases (less data is
  easier to interpolate).
- (ii) in any TPT-reaching seed, NC1_z decays >=1 order of magnitude
  (ratio <=0.1), matching every prior TPT-gated arm (Stage 12 P1, Stage 15
  HN3).
- (iii) reupload=false total_var invariance holds at machine precision
  (max relative change <1e-6 at every checkpoint) regardless of whether
  TPT is reached -- T5 is a property of the encoding structure, not of
  training outcome, per Stage 19 V1(i)'s replication on real data.

Scoring: (i) CONFIRMED if the spc=50 TPT fraction >= the spc=100 TPT
fraction in both reupload arms (or both are 0, which is consistent with
"increases or stays at floor" and reported as such, not scored as
REFUTED); PARTIAL if the direction holds in only one reupload arm;
REFUTED if spc=100's fraction exceeds spc=50's in either arm. (ii)
scored only on TPT-reaching seeds if any exist; UNSCOREABLE (as in Stage
19) if none reach TPT -- reported factually regardless. (iii) CONFIRMED
if 5/5 (or however many seeds actually run) reupload=false seeds hold the
threshold; REFUTED on any single violation (no partial credit, matching
T5's exact-invariance claim throughout this codebase).

### Arm W2 -- contraction-vs-rotation contrast (the headline comparison)

New metric (SPEC3_ADDENDUM.md section 24): `within_class_scatter_trace` =
tr(Sigma_W), the ABSOLUTE (unnormalized) within-class scatter trace --
the direct classical analogue of `total_var`, as opposed to
`classical_nc1`'s normalized tr(Sigma_W @ pinv(Sigma_B)) ratio. Logged
per epoch for classical MLP runs only (VQC runs already log `total_var`
per checkpoint via the existing reanalyze.py walk).

Runs: reuse Stage 19 Arm V2(a)'s classical MLP architecture and
hyperparameters (4->64->16->C, SGD) but rerun 3 seeds x 2 datasets (6 runs,
existing configs `configs/stage19_classical_{mnist,fmnist}_pca4.yaml` plus
the new logging code) since the original Stage 19 runs did not persist raw
features/scatter traces (checkpoints exist but are not sufficient without
re-deriving the metric; a clean rerun is simpler and cheap at 600 epochs).
VQC side reuses Stage 19 Arm V2(b)'s (no-reupload, R0, matched PCA-4)
existing checkpoints -- no VQC rerun.

Predicted: classical tr(Sigma_W) contracts materially (final/initial ratio
< 0.5) in all 6 seeds, while the matched fixed-encoding VQC's total_var
stays invariant at ~1e-15 (machine precision, matching V1(i)/V2(b)) on the
same PCA-4 inputs.

Scoring: CONFIRMED if the classical ratio < 0.5 in >=5/6 seeds AND the VQC
total_var_max_rel_change < 1e-6 in all reused seeds; PARTIAL if the
classical ratio holds in 4/6 seeds or one dataset only; REFUTED otherwise.

Deliverable: two-panel figure `figs/fig19_contraction_vs_rotation.png`,
classical tr(Sigma_W) vs epoch beside VQC total_var vs epoch, same axis
style, one exemplar seed per panel (or small-multiples across seeds if
space allows).

### Arm W3 -- presentation fixes (no new runs, no predictions)

Not a scored arm -- a documentation/audit task. Two deliverables:

1. Split Stage 19's Arm V2 comparative table (`STAGE19_FINDINGS.md`) into
   two separate per-dataset blocks (MNIST, Fashion-MNIST), no pooling
   across datasets, plus a new column for absolute within-class
   contraction (tr(Sigma_W) final/init, from Arm W2 where available;
   marked N/A for approaches Arm W2 does not rerun).
2. Full preregistration ledger recount across ALL stages (12 through 20),
   from scratch, with an explicit counting-unit convention stated BEFORE
   totalling (chosen now, not fitted post-hoc to match any prior claimed
   total): **count each independently-verdicted leg as one item** -- i.e.
   if a stage's own findings file assigns separate CONFIRMED/REFUTED/
   PARTIAL verdicts to sub-parts of one preregistration header (e.g. Stage
   19's V2(d), scored full-space CONFIRMED + z-space REFUTED separately;
   or HR6/HN5, one ledger row covering two registered predictions), each
   independently-verdicted leg counts as one item, not the header as a
   whole. This convention is applied uniformly to every stage, which may
   revise historical single-header counts, not just add Stage 19/20's.
   The recount is reconciled against the paper's current claimed figure
   of 35 (`paper_ieee/NUMBER_AUDIT_IEEE.md` line 47: "35 preregistered
   items: 23 CONFIRMED, 6 REFUTED, 5 PARTIAL, 1 reported factually",
   itself built from `RESULTS_V3.md`'s 30-item tally plus Stage 18's
   B1-B5), with any discrepancy flagged explicitly and its arithmetic
   shown, not just a corrected number asserted.

### Arm W4 -- amplitude-encoding probe (registered addition)

Diagnostic reasoning: at n=4, L=16 is M=192 against a DLA bound of 255 (M/
M_c ~ 0.75), and Stage 19's L=32 capacity-check probe was M=384 (~1.5x the
bound) -- both far past the M/M_c~0.376 where Phase 2's TPT transition
completed (RESULTS_V2.md section 4). The vision runs plateau despite this
headroom, while a capacity-matched classical MLP interpolates the
identical PCA-4 features in 5/6 seeds (Stage 19 V2(a)). This combination
points away from variational capacity and toward the angle encoding
itself: compressing the input into 4 rotation angles may not span a state
manifold that separates these classes, independent of circuit depth.

Amplitude encoding (`encoding: amplitude`, already implemented, Stage 10
bars-and-stripes) tests this directly without threatening T5: the theorem
only requires the encoding be applied once and be parameter-independent,
which `qml.AmplitudeEmbedding` satisfies identically to angle encoding --
U(theta) still acts as a single global unitary on a fixed constellation.

Probe: MNIST classes {0,1,2}, PCA-16 (not PCA-4) mapped to normalized
amplitudes on n=4 qubits (2^4=16, using all 16 PCA dimensions rather than
4), samples_per_class=50, reupload in {true, false}, 2 seeds each (0-1) =
4 runs, L=16, R0. Requires a new no-angle-rescale PCA path in `data.py`
(existing `pca_encode` hardcodes `n_components=min(n_qubits,raw_dim)` and
rescales to [-pi,pi] for RY rotations, both wrong for amplitude
embedding, which wants the raw/standardized 16-d vector L2-normalized by
`AmplitudeEmbedding(normalize=True)` itself).

Predicted: no directional prediction on TPT (explicitly a scoping probe,
not a hypothesis test) -- every outcome is reportable. If TPT is reached
in >=1 seed, that is the real-data terminal-phase point missing from this
paper to date. If TPT is not reached in 0/4, that is a cleaner, stronger
scoping statement ("the gap survives an encoding change") than "angle
encoding failed" alone. Separately, reupload=false total_var invariance
(<1e-6 max relative change) IS predicted to hold under amplitude encoding
too, since T5's proof does not reference the encoding family.

Scoring: TPT fraction reported factually, ungated, either way (not
CONFIRMED/REFUTED -- there is no directional claim to falsify). The
total_var invariance sub-claim is scored CONFIRMED/REFUTED on the usual
<1e-6 threshold, independently of the TPT outcome.

### Interpretation gate (Stage 20)

W1, W2, W4 scored independently against the thresholds above in
`STAGE20_FINDINGS.md`, exactly as every prior stage's arms have been. No
softening after the fact (CLAUDE.md). W3 is a presentation/audit
deliverable with no predictions to score. All trim/fallback rules above
are fixed at this commit; any deviation after seeing probe output is
reported as a deviation, not silently applied.
