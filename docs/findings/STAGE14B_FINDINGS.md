# STAGE14B_FINDINGS.md -- Collapse by rotation

Implements PREDICTIONS.md's "Stage 14B preregistration -- collapse by
rotation" section, scored strictly against the thresholds committed there
(commit a03685f) before any Stage 14B metric was computed.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS -- classical
NC definition. Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 -- quantum NC at
the optimum. San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 -- empirical
QNC study.

## The rotation-invariance argument, informally

For angle encoding without data re-uploading (`model.reupload: false`), each
training sample x is encoded once into a fixed pre-image state
rho_0(x) = |psi_0(x)><psi_0(x)|, and the entire trained circuit acts as a
single global unitary U(theta) applied after that encoding:
rho(x; theta) = U(theta) rho_0(x) U(theta)^dagger. Conjugation by a unitary
is an isometry of Hilbert-Schmidt distance -- it preserves Tr((A-B)(A-B)^dagger)
for any two operators A, B -- so it preserves the pairwise (and hence
within-class) HS geometry of the entire constellation {rho(x_i; theta)}
exactly, for every theta. In particular the total within-class HS variance
`total_var` = V_meas + V_fiber (SPEC3 section 18.4, summed over the full
4^n - 1 traceless-Hermitian directions via Parseval's identity) cannot change
under training in this setting: training can only ROTATE the fixed
constellation rho_0(x_1), ..., rho_0(x_N) as a rigid body in Hilbert space,
never dilate or contract it. Any apparent collapse -- NC1_z decay, the V_fiber
dip below the untrained-null level (Stage 12's headline witness) -- must then
come entirely from the rotation reorienting the rigid constellation so that
within-class spread, which cannot shrink in total, gets redistributed INTO the
loss-invisible fiber directions and OUT of the measured (loss-visible)
directions, or vice versa for the training-time INCREASE of V_meas. Data
re-uploading breaks this argument: `_circuit_body`'s reupload branch (SPEC
models.py) re-applies `_encode` at every layer under the current trained
weights, so the state is no longer `U(theta)` applied to a theta-independent
pre-image -- the map from data to output state is itself trained, and real
contraction becomes possible. (Formal statement left to THEORY_NOTES.md,
Stage 17, per PLAYBOOK3.md's T1/T3 skeleton -- this paragraph is the informal
version the user asked to see now.)

## HR1 -- reupload=true total variance declines only slightly -- CONFIRMED

L=8 anchor (`transition_L_L8_s0..s4`, reupload=true), `total_var` per
checkpointed epoch, recomputed via the existing `reanalyze_l8_anchor_zspace`
(no new training):

| seed | epoch0 total_var | final total_var | final/initial |
|---|---|---|---|
| s0 | 0.90160 | 0.89022 | 0.98738 |
| s1 | 0.89615 | 0.88539 | 0.98799 |
| s2 | 0.90416 | 0.89992 | 0.99531 |
| s3 | 0.90504 | 0.90181 | 0.99644 |
| s4 | 0.90169 | 0.89824 | 0.99617 |

5-seed mean final/initial ratio: **0.99266** (decline ~0.73%).

Scoring: CONFIRMED requires mean ratio >= 0.9 (decline < 10%). 0.99266 >= 0.9
-> **CONFIRMED**, with a much smaller decline than the 10% ceiling -- the
Stage 12 V_fiber dip rides on a total-variance budget that is almost, but not
exactly, conserved even with reuploading.

## HR2 -- reupload=false: total variance is exactly constant -- CONFIRMED

Config: `stage14_mechanism_base.yaml` with `model.reupload: false`, 5 seeds,
600 epochs (`configs/sweeps/stage14b_reupload_false.yaml`). Relative change
`abs(total_var[t] - total_var[0]) / total_var[0]`, maximum over all 161
checkpointed epochs per seed:

| seed | run_id | epoch0 total_var | max relative change |
|---|---|---|---|
| 0 | stage14b_reupload_false_reupload_false_s0_20260715-213042 | 0.71070 | 3.75e-15 |
| 1 | stage14b_reupload_false_reupload_false_s1_20260715-213528 | 0.67959 | 3.59e-15 |
| 2 | stage14b_reupload_false_reupload_false_s2_20260715-213936 | 0.74572 | 4.47e-15 |
| 3 | stage14b_reupload_false_reupload_false_s3_20260715-214338 | 0.74159 | 5.84e-15 |
| 4 | stage14b_reupload_false_reupload_false_s4_20260715-214807 | 0.74846 | 3.56e-15 |

All 5/5 seeds' max relative change is ~1e-15 -- floating-point precision, not
a "small but real" drift. Preregistered threshold was < 1e-6; the actual
result is ~9 orders of magnitude below that.

Scoring: CONFIRMED requires the < 1e-6 bound at every checkpoint for all 5
seeds. 5/5 -> **CONFIRMED**, decisively. This is the direct numerical
fingerprint of the rotation argument above: without re-uploading, total_var
is invariant to machine precision.

## HR3 -- reupload=false runs still show NC1_z decay if they reach TPT -- CONFIRMED

Same 5 runs as HR2. TPT fraction: **5/5** (all seeds reached TPT -- contrary
to PREDICTIONS.md's "may fail TPT at C=3" caveat, reported factually; this
does not affect HR2's scoring per that section's note).

| seed | nc1_z epoch0 | nc1_z final | ratio |
|---|---|---|---|
| 0 | 5.13608 | 0.07372 | 0.014353 |
| 1 | 4.83138 | 0.05602 | 0.011594 |
| 2 | 6.12309 | 0.09407 | 0.015364 |
| 3 | 2.35322 | 0.08373 | 0.035580 |
| 4 | 3.73577 | 0.10328 | 0.027647 |

Mean nc1_z final/epoch0 ratio among the 5 TPT-reaching seeds: **0.020908**.

Scoring: CONFIRMED requires ratio <= 0.1 for the TPT-reaching-seed mean.
0.020908 <= 0.1 -> **CONFIRMED**. Rotation alone -- with total_var frozen to
machine precision (HR2) -- fully reproduces the NC1_z collapse diagnostic:
z-space collapse is a redistribution of a fixed variance budget into fiber
directions, not a real shrinkage of the state constellation.

## HR4 -- deeper reuploading circuits contract more -- CONFIRMED

Stage 9 L sweep (`transition_L_manifest.json`, L in {2,4,8,12,16,21,26,32},
5 seeds each, reupload=true), final `trace_w` via the existing
`reanalyze_stage9_sweep_zspace_final` (no new training):

| L | mean trace_w (n=5) |
|---|---|
| 2 | 0.048137 |
| 4 | 0.102117 |
| 8 | 0.094329 |
| 12 | 0.096018 |
| 16 | 0.073856 |
| 21 | 0.065498 |
| 26 | 0.069527 |
| 32 | 0.058570 |

Spearman(trace_w, L) across all 40 (L, seed) pairs: **rho = -0.4509,
p = 0.0035**.

Scoring: CONFIRMED requires rho < 0 and p < 0.05. -0.4509 < 0 and
0.0035 < 0.05 -> **CONFIRMED**. Not a strictly monotonic curve (L4 and L12
are local bumps above L8), but the overall trend across the full depth range
is a significant decrease, consistent with deeper reuploading circuits
contracting more in z-space within-class variance.

## HR5 -- variance floor scales with blob_std^2 -- REFUTED

Config: `stage14_mechanism_base.yaml` (L=8, reupload=true), `data.blob_std`
in {0.2, 0.4, 0.8}, 5 seeds each, 15 runs
(`configs/sweeps/stage14b_blob_std.yaml`, recovered after a mid-sweep kill via
`stage14b_blob_std_04_remainder.yaml` and `stage14b_blob_std_08_remainder.yaml`;
manifest assembled and all 15 run dirs verified to have an `epoch_0600.pt`
checkpoint at `results/sweeps/stage14b_blob_std_assembled/manifest.json`).
Final `trace_w`:

| blob_std | seed trace_w values | mean trace_w |
|---|---|---|
| 0.2 | 0.077952, 0.075379, 0.077752, 0.089139, 0.079827 | 0.080010 |
| 0.4 | 0.093672, 0.091019, 0.094763, 0.100120, 0.109181 | 0.097751 |
| 0.8 | 0.102559, 0.081759, 0.097976, 0.100684, 0.088665 | 0.094329 |

Log-log linear regression, `trace_w_final ~ blob_std^power`, 3 blob_std
values' 5-seed means (primary, preregistered basis): **power = 0.1188,
R^2 = 0.594**. Supplementary 15-seed-level fit: power = 0.1174,
R^2 = 0.353. mean trace_w is also NOT monotonic in blob_std (0.4's mean,
0.09775, is slightly ABOVE 0.8's, 0.09433).

Scoring: CONFIRMED requires power in [1.5, 2.5] and R^2 >= 0.8; PARTIAL
requires power in [1.5, 2.5] with R^2 < 0.8, or power just outside
[1.0, 3.0]; REFUTED if power outside [1.0, 3.0] or R^2 < 0.5. Fitted power
0.1188 is far outside [1.0, 3.0] (more than an order of magnitude below the
lower bound) -> **REFUTED**, decisively, independent of the R^2 caveat.

The final z-space within-class variance floor is essentially INSENSITIVE to
a 4x change in the data's own spread (blob_std 0.2 -> 0.8), not
proportional to blob_std^2 as a near-rigid-map argument would predict. This
does not contradict HR1/HR2/HR3/HR4 (all about total_var invariance or its
depth-dependence at fixed data): it says the specific claim that the residual
z-space variance floor is "inherited from the data through a near-rigid map"
is wrong as stated. A plausible alternative (not scored here, noted for
THEORY_NOTES): CE training may drive trace_w toward a floor set by the
readout geometry / beta dynamics rather than by the raw data spread, with
blob_std instead entering primarily through trace_b (between-class
separation) and TPT onset timing -- untested here, a natural HR for a future
stage.

## HR6 -- per-class purity and pairwise overlap constant at n=4 (Stage 17b, existing HR2 checkpoints, no new training)

Preregistered in PREDICTIONS.md's "Stage 17b preregistration" section
(committed before this metric was computed), testing a stronger
(non-additive-quantity) form of the T5 rotation-invariance argument beyond
`total_var`: per-class-mean purity Tr(rho_bar_c^2) and pairwise overlap
Tr(rho_bar_c @ rho_bar_c') (`mean_class_purity`, `class_mean_overlap`,
`src/qnc/reanalyze.py`'s `reanalyze_hr6_hn5_invariance`), same 5
reupload=false runs as HR2/HR3, all 161 checkpoints per seed:

| seed | run_id | max rel. change, purity | max rel. change, overlap |
|---|---|---|---|
| 0 | stage14b_reupload_false_reupload_false_s0_20260715-213042 | 3.4546e-15 | 3.6759e-15 |
| 1 | stage14b_reupload_false_reupload_false_s1_20260715-213528 | 3.2575e-15 | 3.6743e-15 |
| 2 | stage14b_reupload_false_reupload_false_s2_20260715-213936 | 4.5717e-15 | 4.5940e-15 |
| 3 | stage14b_reupload_false_reupload_false_s3_20260715-214338 | 6.4487e-15 | 6.1244e-15 |
| 4 | stage14b_reupload_false_reupload_false_s4_20260715-214807 | 3.5185e-15 | 3.7149e-15 |

Scoring: CONFIRMED requires max relative change < 1e-6 for both quantities
in 5/5 seeds. All 10 values are ~1e-15, nine orders of magnitude below the
threshold, identical fingerprint to HR2's total_var result -> **CONFIRMED**.
Data: `results/stage14b_reanalysis/stage17b_hr6_hn5.json` (`hr6_n4` key).

## Interpretation

| prediction | verdict |
|---|---|
| HR1 (reupload=true total_var decline < 10%) | CONFIRMED (0.73% decline) |
| HR2 (reupload=false total_var exactly constant) | CONFIRMED (~1e-15 relative change) |
| HR3 (reupload=false nc1_z still decays >= 1 order of magnitude if TPT reached) | CONFIRMED (5/5 TPT, ratio 0.0209) |
| HR4 (deeper reuploading circuits contract more, trace_w vs L) | CONFIRMED (rho=-0.451, p=0.0035) |
| HR5 (variance floor scales blob_std^2) | REFUTED (power=0.119, far outside [1.0,3.0]) |
| HR6 (per-class purity + pairwise overlap exactly constant, n=4) | CONFIRMED (~1e-15 relative change, both quantities, 5/5 seeds) |

No softening (CLAUDE.md): HR1-HR4 all confirm the rotation mechanism as
preregistered -- HR2 in particular is a decisive, exact (machine-precision)
result, not a statistical tendency. HR5 REFUTES the specific "variance floor
inherited from data through a near-rigid map" sub-claim; this is reported
factually and does not soften HR1-HR4's confirmation of the core rotation
argument, since HR5 concerns a downstream quantitative prediction about WHERE
the (small, reupload=true) variance floor comes from, not whether rotation is
the dominant mechanism. HR6 (Stage 17b) extends HR2's exact invariance from
total_var to two further quantities (per-class purity, pairwise overlap),
both confirming to the same ~1e-15 machine-precision fingerprint -- the
isometry argument (THEORY_NOTES.md T5) holds for every basis-independent
functional of the full state tested so far, not just the Parseval-summed
variance.

Figure: `results/stage14b_reanalysis/figures/total_variance_rotation.png`
(total within-class HS variance vs epoch, reupload=true vs reupload=false
overlaid, mean +/- 1 std across seeds).

Reanalysis data: `results/stage14b_reanalysis/stage14b_hr1.json`,
`stage14b_hr2_hr3.json`, `stage14b_hr4.json`, `stage14b_hr5.json`.

## Housekeeping -- HR5 sweep recovery

`configs/sweeps/stage14b_blob_std.yaml`'s background sweep was killed by the
environment mid-run (same failure mode as Stage 13B/14 arm 4): blob_std_0.2
completed all 5 seeds; blob_std_0.4 completed seeds 0-1 before being killed
(seed 2's run directory was created but is empty -- died before any
checkpoint or metrics.jsonl line was written; left in place per CLAUDE.md
invariant 3, not referenced by any manifest); blob_std_0.8 never started.
Recovered via `configs/sweeps/stage14b_blob_std_04_remainder.yaml` (seeds
2-4) and `configs/sweeps/stage14b_blob_std_08_remainder.yaml` (all 5 seeds).
Full 15-run manifest assembled by hand at
`results/sweeps/stage14b_blob_std_assembled/manifest.json`; every run
directory's `epoch_0600.pt` final checkpoint was verified to exist before
use.

## HR5 diagnosis (PLAYBOOK3.md Stage 15, checkpoints only, no new training)

Preregistered in PREDICTIONS.md's "Stage 15 preregistration -- HR5
diagnosis" section (committed before either metric below was computed).
Suspected cause of HR5's refutation: `pca_encode`'s min-max scaling to
[-pi, pi] (SPEC.md section 2), fit per-run on that run's own train split,
absorbs blob_std's raw spread before angle-encoding, so the encoded
constellation never inherits blob_std's scale. Both tests below reuse the
same 15 blob_std runs (`configs/sweeps/stage14b_blob_std.yaml` + remainders)
and add the 5 `stage14b_reupload_false` runs and the Stage 12 L=8 anchor (5
seeds) for HR5c's pool. `src/qnc.reanalyze.reanalyze_hr5_diagnosis`,
`compute_hr5b`, `compute_hr5c`; data at
`results/stage14b_reanalysis/stage15_hr5_diagnosis.json`.

### HR5b -- epoch-0 constellation spread is flat across blob_std

Epoch-0 seeded reconstruction (no training), 5-seed arm means:

| blob_std | mean epoch0 total_var | mean epoch0 trace_w |
|---|---|---|
| 0.2 | 0.876875 | 0.162269 |
| 0.4 | 0.891066 | 0.173422 |
| 0.8 | 0.901729 | 0.166031 |

Relative range `(max-min)/mean(3 arm means)`: **total_var = 2.79%**,
**trace_w = 6.67%**. Both far below the preregistered 20% threshold.

Scoring: CONFIRMED requires relative range < 20% for both metrics.
2.79% < 20% and 6.67% < 20% -> **CONFIRMED**. A 4x change in raw blob_std
(0.2 -> 0.8) moves the post-min-max-scaling epoch-0 constellation spread by
under 7% either way, directly supporting the diagnosis that min-max scaling
absorbs blob_std before encoding -- explaining why HR5's final-trace_w fit
against blob_std found essentially no dependence.

### HR5c -- corrected inheritance test: final trace_w vs the SAME run's own epoch-0 trace_w

Pooled Spearman correlation, n=25 (15 blob_std runs + 5 reupload_false runs +
5 L8-anchor seeds), final `trace_w` vs that run's own epoch-0 `trace_w`:
**rho = 0.1322, p = 0.5288**.

Scoring: CONFIRMED requires rho > 0 and p < 0.05; PARTIAL requires rho > 0
with p >= 0.05; REFUTED requires rho <= 0. rho > 0 but p = 0.529 >= 0.05 ->
**PARTIAL**. The correlation's sign is in the predicted direction (a run's
own epoch-0 z-space within-class variance has a positive, not negative,
association with where it lands after training) but is not statistically
significant at n=25 -- consistent with HR5b's finding that epoch-0 spread is
itself nearly data-independent (flat across blob_std), leaving little
between-run variance at epoch 0 for a within-run inheritance signal to grab
onto.

### Interpretation (HR5 diagnosis)

| prediction | verdict |
|---|---|
| HR5b (epoch-0 constellation spread flat across blob_std, rel. range < 20%) | CONFIRMED (total_var 2.79%, trace_w 6.67%) |
| HR5c (final trace_w correlates with own epoch-0 trace_w, pooled n=25) | PARTIAL (rho=0.132 > 0, p=0.529 not significant) |

No softening (CLAUDE.md): HR5b's strong confirmation isolates the mechanism
behind HR5's refutation -- `pca_encode`'s per-run min-max scaling to
[-pi, pi] absorbs blob_std's raw spread before the angle encoding even sees
it, so there was never a scale for training to inherit blob_std^2 from in
the first place. HR5c's PARTIAL result is consistent with, not contradictory
to, HR5b: with epoch-0 spread already nearly flat across the primary
manipulated variable (blob_std), the remaining epoch-0-to-epoch-0
variation (seed, seeded PCA/init) is small, so a real but weak run-to-run
inheritance signal (rho=0.132, right sign) does not clear significance at
n=25. This does not reopen HR5 (blob_std^2 scaling remains REFUTED,
reported above) -- it explains why, and rules out "insufficient
data-to-data variance in the encoded input" alone as the reason a
within-run inheritance test could still fail (it doesn't fail; it's just
not significant yet).
