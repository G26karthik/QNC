# STAGE14_FINDINGS.md -- Mechanism tests of the scaling-escape hypothesis

Implements PLAYBOOK3.md Stage 14 / SPEC3_ADDENDUM.md section 20, as reframed
by this stage's binding task prompt: the six arms below are MECHANISM TESTS
of the scaling-escape hypothesis (NC1_z decay is driven by between-class
separation growth, tr(Sigma_B^z), while within-class variance tr(Sigma_W^z)
barely shrinks; learnable beta is hypothesized to be the escape valve that
lets training avoid full within-class collapse). Scored strictly against
PREDICTIONS.md's "Stage 14 preregistration" section, committed (34b6b2a)
before any Stage 14 metric below was computed.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS -- classical
NC definition. Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 -- quantum NC at
the optimum. San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 -- empirical
QNC study.

## Baseline

L=8 R0 baseline (Stage 13 matched-M sweep's R0 arm,
`results/sweeps/stage13_expressivity_20260714-111146/`, 5 seeds) recomputed
fresh here (`reanalyze.reanalyze_stage14_baseline`) rather than reused from
STAGE12_FINDINGS.md, for an apples-to-apples comparison against the new
arms (same `stage13_expressivity.yaml` base config as arms 1/3/6).

## Witness-set table (arms 1, 3, 6 vs baseline; final checkpointed epoch)

| arm | trace_w | trace_b | nc1_z | concentration_factor |
|---|---|---|---|---|
| R0_baseline | 0.0943287 +/- 0.00789311 | 0.108972 +/- 0.0175061 | 0.279491 +/- 0.0658346 | 1 +/- 0 |
| beta_max500 | 0.0943287 +/- 0.00789311 | 0.108972 +/- 0.0175061 | 0.279491 +/- 0.0658346 | 1 +/- 0 |
| label_smoothing | 0.0897048 +/- 0.00609537 | 0.0920131 +/- 0.0142667 | 0.275037 +/- 0.0443095 | 1 +/- 0 |
| frozen_beta | 0.108429 +/- 0.0063976 | 0.16468 +/- 0.0365408 | 0.236752 +/- 0.0364834 | 1 +/- 0 |

Mann-Whitney U, R0_baseline vs each other arm:

| arm | trace_w p-value | trace_b p-value | nc1_z p-value | concentration_factor p-value |
|---|---|---|---|---|
| beta_max500 | 1 | 1 | 1 | 1 |
| label_smoothing | 0.4206 | 0.09524 | 0.6905 | 1 |
| frozen_beta | 0.01587 | 0.01587 | 0.2222 | 1 |

`concentration_factor` = 1.0 exactly (0 std) for every seed of every arm --
the pipeline sanity check (PREDICTIONS.md's degenerate-R0 note) is MET, no
recompute-pipeline bug.

epoch-0 -> final decay ratios (mean, 5 seeds each):

| arm | nc1_z epoch0 | nc1_z final | nc1_z ratio | trace_w epoch0 | trace_w final | trace_w ratio |
|---|---|---|---|---|---|---|
| R0_baseline | 62.06 | 0.2795 | 0.004504 | 0.166 | 0.09433 | 0.5681 |
| beta_max500 | 62.06 | 0.2795 | 0.004504 | 0.166 | 0.09433 | 0.5681 |
| label_smoothing | 62.06 | 0.2750 | 0.004432 | 0.166 | 0.08970 | 0.5403 |
| frozen_beta | 62.06 | 0.2368 | 0.003815 | 0.166 | 0.10843 | 0.6531 |

## Arm 1 -- beta_max = 500 (5 seeds) -- CONFIRMED

beta_max500's final `trace_w`, `trace_b`, `nc1_z` are numerically IDENTICAL
(bit-for-bit, not just statistically indistinguishable) to the baseline for
every seed (p=1 on all three fields). Diagnosis, not a pipeline bug: the
baseline's own trained beta never exceeds ~30 (max logged beta per seed:
29.97, 23.80, 25.79, 29.13, 29.88), well under the ORIGINAL 50 clamp, so
raising the ceiling to 500 changes nothing about the optimization
trajectory -- the clamp was never binding to begin with.

- Sub-prediction (a), trace_w weaker-or-equal shrink: arm1 mean == baseline
  mean (equal case) and p=1 (not significant) -- MET.
- Sub-prediction (b), nc1_z decays >= 1 order of magnitude from its own
  epoch-0 value: ratio 0.004504 << 0.1 -- MET.

Both sub-predictions hold -> **CONFIRMED**. Caveat: this arm is scientifically
uninformative about the beta-ceiling's causal role, since the ceiling was
never approached; it does confirm separation-driven NC1_z decay is robust to
a much larger nominal ceiling, which is the prediction as preregistered.

## Arm 2 -- long horizon, 5000 epochs, seed 0 -- CONFIRMED

Run `stage14_long_horizon_s0_20260714-212225`: 5000 epochs completed, no
NaN loss at any of the 101 checkpointed epochs (every 50 epochs), final
epoch loss=0.002066, acc=1.0, `e0_epoch`=35, `tpt_reached`=true.

Spearman(`trace_w`, log(epoch)) over all 100 checkpoints strictly after
e0_epoch=35: rho = -0.0716, p = 0.4793.

Scoring: CONFIRMED if p >= 0.05, or p < 0.05 with rho >= 0; REFUTED if
rho < 0 and p < 0.05. p=0.4793 >= 0.05 -> **CONFIRMED** (no significant
downward drift in trace_w across 5000 epochs -- the Stage 12/13 600-epoch
null extends to a much longer horizon).

Figure: `results/stage14_long_horizon_aggregate/figures/trace_w_drift.png`.

## Arm 3 -- label smoothing 0.1 (5 seeds) -- REFUTED

`trace_w` final mean is lower than baseline in the predicted direction
(0.08970 vs 0.09433, ratio 0.5403 vs 0.5681 -- a real but small additional
shrink), but Mann-Whitney U p=0.4206 >= 0.05 -- not significant.

Scoring requires BOTH p < 0.05 AND arm3 mean < baseline mean (no PARTIAL for
this arm). Direction is right; significance is not met -> **REFUTED**. Label
smoothing's classical NC-acceleration effect (Papyan/Han/Donoho 2020) is not
statistically detectable in within-class z-space variance at this sample
size (n=5 seeds) in the quantum setting.

## Arm 4 -- transition localization, L in {3, 5, 6} (10 seeds each) -- CONFIRMED

Original sweep (`configs/sweeps/stage14_transition_localization.yaml`) was
killed by the environment after L3 (10/10 seeds) and L5 (9/10 seeds)
completed; L5 seed 9 died mid-training (epoch 850 of its extended
`target_epochs`=1200, dir
`results/stage14_transition_localization_L5_s9_20260715-004730`, left in
place per CLAUDE.md invariant 3, not used) and L6 (all 10 seeds) never
started -- the same failure mode as Stage 13B's fixed-depth sweep. Recovered
with two small remainder sweeps,
`configs/sweeps/stage14_transition_localization_l5_remainder.yaml` (L5 seed
9 only) and `configs/sweeps/stage14_transition_localization_remainder.yaml`
(L6, all 10 seeds). Full 30-run manifest assembled by hand:
`results/sweeps/stage14_transition_localization_assembled/manifest.json`
(all 30 run directories verified to exist before use).

| L | TPT fraction (n=10 seeds) |
|---|---|
| L3 | 0.3 |
| L5 | 1.0 |
| L6 | 1.0 |

Scoring: CONFIRMED if TPT-fraction is non-decreasing L3 -> L5 -> L6; PARTIAL
if non-monotonic but L6 > L3; REFUTED if L6 <= L3. 0.3 -> 1.0 -> 1.0 is
non-decreasing -> **CONFIRMED**. L5 and L6 both reach TPT in all 10 seeds
(no further separation between them at these two depths); L3 reaches TPT in
only 3/10 seeds, consistent with the original SPEC3 section 20 prediction
and the Phase 2 "TPT reached somewhere between 9% and 38% of M_c" framing.

Figure: `results/sweeps/stage14_transition_localization_assembled/figures/tpt_fraction.png` (written by `make_stage14_tpt_fraction_figure`, saved directly to `results/sweeps/stage14_transition_localization_assembled/figures/tpt_fraction.png`).

## Arm 5 -- single-input QFI rank fix, L in {8, 16, 21, 26, 32} -- REFUTED

No new training (epoch-0 seeded reconstruction, seed 0, off the existing
Stage 9 `transition_L` configs).

| L | single-input ranks (4 inputs) | mean single-input rank | batch-16 epoch-0 rank (original Stage 9) |
|---|---|---|---|
| L8 | 30, 30, 30, 30 | 30.0 | 96 |
| L16 | 30, 30, 30, 30 | 30.0 | 192 |
| L21 | 30, 30, 30, 30 | 30.0 | 252 |
| L26 | 30, 30, 30, 30 | 30.0 | 312 |
| L32 | 30, 30, 30, 30 | 30.0 | 384 |

Scoring: CONFIRMED if mean single-input rank at L=26 AND L=32 is
>= 0.95*255=242; PARTIAL if true at only one; REFUTED if neither. Both are
30.0 -> **REFUTED**, decisively (not close to threshold).

Diagnosis: the single-input rank saturates at EXACTLY 30 = 2*(2^4 - 1) =
2*(dim-1) for every L and every one of the 4 tested inputs, independent of
circuit depth. This is the real dimension of CP^15, the pure-state manifold
for a 4-qubit (dim=16) Hilbert space -- a single input pins the model to one
trajectory through state space, and no matter how many circuit parameters
feed that trajectory, the QFI rank cannot exceed the state manifold's own
real dimension. The preregistered prediction (near-255 saturation, closing
a "batch aggregation under-reports" methodological gap) is refuted, but for
a well-understood structural reason, not a bug: single-input QFI rank is
bounded by the state-manifold dimension (30), not the circuit's raw
parameter-space bound (4^n - 1 = 255) that the batch-aggregated rank
approaches. The batch-aggregated rank's growth with L (96/192/252/312/384,
capped in practice by min(M, 255)) reflects that averaging over multiple
inputs samples multiple tangent directions on the state manifold; a single
input cannot.

## Arm 6 -- frozen beta, beta=5.0 non-learnable (5 seeds) -- REFUTED

`trace_w` final mean is HIGHER than baseline, not lower (0.10843 vs 0.09433,
ratio 0.6531 vs 0.5681 -- weaker shrink, opposite of the predicted
direction), Mann-Whitney U p=0.01587 (significant, but in the wrong
direction).

Scoring requires p < 0.05 AND arm6 mean < baseline mean (no PARTIAL). p is
significant but the direction is reversed -> **REFUTED**. This directly
contradicts the "beta as escape valve" mechanism story: removing beta's
freedom to grow does NOT force the model to fall back on shrinking
within-class z-space variance -- if anything, within-class variance shrinks
LESS without a learnable beta.

E0 (TPT onset), reported factually per PREDICTIONS.md (not itself scored):

| seed | baseline e0_epoch | frozen_beta e0_epoch |
|---|---|---|
| 0 | 35 | 246 |
| 1 | 27 | 60 |
| 2 | 45 | 142 |
| 3 | 32 | 162 |
| 4 | 37 | 116 |

All 5 frozen_beta seeds reach TPT (tpt_reached=true), but substantially
later than baseline (mean E0 145.2 vs 35.2) -- consistent with the
preregistered "E0 later" alternative outcome.

## Interpretation

| arm | verdict |
|---|---|
| 1 (beta_max=500) | CONFIRMED (uninformative: baseline never approached the original clamp) |
| 2 (long horizon, 5000 epochs) | CONFIRMED |
| 3 (label smoothing 0.1) | REFUTED (right direction, not significant) |
| 4 (transition localization L3/5/6) | CONFIRMED |
| 5 (single-input QFI rank) | REFUTED (saturates at state-manifold bound 30, not near-255) |
| 6 (frozen beta) | REFUTED (significant, wrong direction) |

No softening (CLAUDE.md): arms 3 and 6, the two arms most directly testing
the scaling-escape mechanism's causal claim (does something OTHER than
separation growth -- label smoothing pressure, or beta's removal -- force
within-class shrink), both REFUTE it. Arm 6 in particular is a direct
constraint: freezing beta does not recover within-class collapse; it delays
TPT and, if anything, leaves MORE within-class variance at convergence than
the learnable-beta baseline. Combined with arm 1's finding that the beta
ceiling is never binding in this regime, the escape-valve mechanism as
specifically stated (beta's freedom to grow is what prevents within-class
collapse) is not supported by these arms. What remains standing across all
six arms: NC1_z decay itself (separation-driven, tr(Sigma_B^z) growth) is
robust -- it holds in the baseline, under a much larger beta ceiling (arm
1), over a 10x longer horizon (arm 2), and regardless of readout-adjacent
config changes (arms 3, 6 all still show >=1-order-of-magnitude nc1_z decay
per the epoch0->final ratio table above, ratios 0.0038-0.0045) -- only the
proposed CAUSAL mechanism for why within-class variance stays roughly flat
(learnable beta as escape valve) is unsupported.

## Housekeeping -- Task C: Stage 13B R2 recovered-seed verification

Verified `results/sweeps/stage13b_fixed_depth_20260714-143839/manifest.json`
R2 seeds 3 and 4 (the two seeds recovered via
`configs/sweeps/stage13b_fixed_depth_r2_remainder.yaml` after the original
sweep was killed):

| seed | run_id | final epoch | n_train_records | final checkpoint | tpt_reached | e0_epoch |
|---|---|---|---|---|---|---|
| 3 | stage13b_fixed_depth_r2_remainder_R2_s3_20260714-142851 | 600 | 161 | epoch_0600.pt | true | 19 |
| 4 | stage13b_fixed_depth_r2_remainder_R2_s4_20260714-143344 | 600 | 161 | epoch_0600.pt | true | 22 |

Both ran the full 600 epochs (matching the base config's `train.epochs`),
have an `epoch_0600.pt` final checkpoint on disk, and their run_ids match
`manifest.json`'s R2 seed-3/seed-4 entries exactly. **Verification PASSES.**

For contrast, the orphaned original R2 seed-3 attempt
(`stage13b_fixed_depth_R2_s3_20260714-133921`, left in place per CLAUDE.md
invariant 3, not referenced by the manifest) stopped at epoch 460/600 --
matching STAGE13B_FINDINGS.md's own account of the kill.

Aside (not required by Task C, noted for completeness): the same fixed-depth
sweep also left two completed-but-unused R1 duplicate run directories
(`stage13b_fixed_depth_R1_s0_20260714-124435`,
`stage13b_fixed_depth_R1_s1_20260714-125227`, both ran the full 600 epochs
but were superseded by later manifest-referenced retries) and one
died-mid-training R1 duplicate
(`stage13b_fixed_depth_R1_s2_20260714-125920`, died at epoch 260/600,
superseded by the manifest-referenced retry). None of these affect
STAGE13B_FINDINGS.md's results (the manifest never referenced them), but
they are orphaned directories on disk per CLAUDE.md's append-only /
never-delete invariant.

## Housekeeping -- Stage 14 arm 4 sweep recovery

Same failure mode as Stage 13B: `configs/sweeps/stage14_transition_localization.yaml`'s
background process was killed mid-run (L3 complete, L5 seed 9 died at epoch
850/1200, L6 never started). Recovered via two remainder sweeps (see Arm 4
section above); the 30-run manifest was assembled by hand and every run
directory's existence was verified before use, matching the Stage 13B R2
recovery precedent.
