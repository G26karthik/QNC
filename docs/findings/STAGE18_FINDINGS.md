# STAGE18_FINDINGS.md — Breadth sprint (external review response)

Citations mandatory: Papyan, Han, Donoho (2020), PNAS (classical NC); Du,
Yang, Tao, Hsieh (2023), PRL 131, 140601 (quantum NC at the static optimum);
San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 (empirical QNC study).

Scored against PREDICTIONS.md's "Stage 18 preregistration" section (B1-B7).
This file is written incrementally as each arm completes; status below
reflects the in-progress state.

## Status

| Arm | Status |
|---|---|
| B1 (optimizer: SGD momentum) | **CONFIRMED** |
| B2 (ansatz: hardware-efficient) | **CONFIRMED** |
| B3 (n=8) | **CONFIRMED** |
| B4 (noisy simulation, inference) | **PARTIAL** |
| B5 (hardware, ibm_kingston) | **CONFIRMED** |
| B6 (statistics upgrade) | done (see STATS_APPENDIX.md) |
| B7 (mechanism schematic) | done (see figs/fig18_mechanism_schematic_*) |

## B1 — optimizer generality: SGD momentum=0.9 (n=4 L=8 R0 reupload=false)

lr-pick (1 seed each, lr in {0.05, 0.1}): both reached TPT (E0=9, E0=8
respectively); lr=0.1 had the lower final loss (0.0012 vs 0.0024) and won
the preregistered tie-break. 5-seed campaign at lr=0.1:

| seed | run_id | TPT | E0 | total_var max rel. change | NC1_z ratio |
|---|---|---|---|---|---|
| 0 | stage18_sgd_campaign_sgd_momentum_lr10_s0_20260718-124505 | True | 8 | 3.28e-15 | 0.0139 |
| 1 | stage18_sgd_campaign_sgd_momentum_lr10_s1_20260718-124918 | True | 9 | 3.43e-15 | 0.0100 |
| 2 | stage18_sgd_campaign_sgd_momentum_lr10_s2_20260718-125340 | True | 6 | 3.57e-15 | 0.0150 |
| 3 | stage18_sgd_campaign_sgd_momentum_lr10_s3_20260718-125759 | True | 9 | 5.54e-15 | 0.0268 |
| 4 | stage18_sgd_campaign_sgd_momentum_lr10_s4_20260718-130220 | True | 12 | 2.82e-15 | 0.0272 |

TPT 5/5; total_var machine-precision invariance holds 5/5 (all <1e-6,
matching HR2's threshold); NC1_z decays >=1 order of magnitude 5/5 (all
ratios <=0.1, max 0.0272). **CONFIRMED** per the preregistered rule
(>=4/5 required). The rotation-invariance mechanism (T5) and the collapse
phenomenology are not Adam-specific.

## B2 — ansatz generality: hardware-efficient ansatz (per-qubit RY-RZ + linear-chain CNOT), n=4 anchor

**Correction (caught during paper integration):** this section previously
described the entangler as "ring CZ." Per `models.py`'s
`_hardware_efficient_layer` (lines 186-192), the actual entangler is a
linear (open-chain) CNOT ladder (`qml.CNOT(wires=[q, q+1])` for
`q in range(n_qubits-1)`, no wraparound), not a ring, and not CZ. The
corrected description is used throughout this section and downstream
(paper_ieee).

**Deviation from the preregistered depth-probe plan (documented per the
interpretation gate, not silently applied):** the depth probe was
preregistered as L in {8, 16, 32, 64}, 3 seeds each. L8 reached 3/3 TPT
(E0=185, 59, 149) and L16 also reached 3/3 TPT (E0=20, 17, 32) — L8 is
therefore already the smallest interpolating depth per the preregistered
selection rule, and continuing to L32/L64 could not change that answer.
The L32 run in progress (seed 1, after seed 0 took ~1h46m vs L8/L16's
~5-10 min) was killed once L8's 3/3 result was confirmed, and L64 was
never started. This mirrors arm B2's own framing ("depth probe **to
interpolation**") rather than Stage 15's precedent of running the full
depth list even after finding the answer early; the deviation is a time/
compute economization, not a change to the scoring rule or its outcome.

Campaign at L8, both arms, 5 seeds each (reupload=true seeds 0-2 reused
from the depth probe per this project's "remainder" convention; seeds 3-4
+ all of reupload=false run fresh):

### reupload=true (5 seeds)

| seed | run_id | TPT | E0 | NC1_z ratio |
|---|---|---|---|---|
| 0 | stage18_hwe_depth_probe_L8_s0_20260718-123715 | True | 185 | 0.0166 |
| 1 | stage18_hwe_depth_probe_L8_s1_20260718-124244 | True | 59 | 0.0124 |
| 2 | stage18_hwe_depth_probe_L8_s2_20260718-124636 | True | 149 | 0.0040 |
| 3 | stage18_hwe_campaign_true_remainder_L8_reupload_true_s3_20260718-174104 | True | 86 | 0.0031 |
| 4 | stage18_hwe_campaign_true_remainder_L8_reupload_true_s4_20260718-174930 | True | 120 | 0.0181 |

TPT 5/5, NC1_z ratio <=0.1 in 5/5 (max 0.0181).

### reupload=false (5 seeds) — headline claim

| seed | run_id | total_var max rel. change |
|---|---|---|
| 0 | stage18_hwe_campaign_false_L8_reupload_false_s0_20260718-175604 | 2.81e-15 |
| 1 | stage18_hwe_campaign_false_L8_reupload_false_s1_20260718-175953 | 3.59e-15 |
| 2 | stage18_hwe_campaign_false_L8_reupload_false_s2_20260718-180334 | 3.28e-15 |
| 3 | stage18_hwe_campaign_false_L8_reupload_false_s3_20260718-180724 | 3.29e-15 |
| 4 | stage18_hwe_campaign_false_L8_reupload_false_s4_20260718-181112 | 3.71e-15 |

Machine-precision invariance holds 5/5 (all <1e-6, same order of
magnitude as HR2/HN1/HR6/HN5's ~1e-15 results at n=4/n=6 under
strongly_entangling). **CONFIRMED** per the preregistered rule (both
bullets hold in 5/5). The rotation-invariance theorem (T5) is a property
of the no-reupload encoding structure, not of the entangling-layer ansatz
family: it replicates exactly under a genuinely different ansatz
(per-qubit RY-RZ + linear-chain CNOT, 2 params/qubit/layer, vs
StronglyEntanglingLayers' ring CNOT and 3 params/qubit/layer).

## B3 — third qubit count: n=8 (blobs3 lifted to R^8)

Depth probe on reupload=true (blob_dist=3.0, blob_std=0.8, 60 samples/class,
600 epochs, same "lift" recipe as Stage 15's n=6 base): L8 reached 3/3 TPT
(E0=40, 38, 24), so per the preregistered trim rule L8 is the one clean
interpolating point and L32 was never run to completion (killed once L8's
answer was confirmed, consistent with B2's identical framing). Campaign
at L8, both arms, 5 seeds each (seeds 0-2 of reupload=true reused from the
timing probe/depth probe per the "remainder" convention).

**Preregistration gap found and closed:** PREDICTIONS.md's B3 entry
predicted reupload=false total_var invariance, but only reupload=true runs
had been launched. 5 fresh reupload=false seeds were added
(`configs/sweeps/stage18_n8_reupload_false.yaml`) to fill this before
scoring.

### reupload=true (5 seeds) — fiber_fraction vs the n=6 null

| seed | run_id | total_var max rel. change | NC1_z ratio | fiber_fraction |
|---|---|---|---|---|
| 0 | stage18_n8_s0_20260718-121735 | 1.50e-04 | 0.0005 | 0.999977 |
| 1 | stage18_n8_depth_probe_l8_remainder_L8_s1_20260718-160647 | 1.84e-04 | 0.0017 | 0.999978 |
| 2 | stage18_n8_depth_probe_l8_remainder_L8_s2_20260718-183839 | 1.93e-04 | 0.0016 | 0.999975 |
| 3 | stage18_n8_campaign_remainder_L8_reupload_true_s3_20260718-194350 | 2.13e-04 | 0.0004 | 0.999975 |
| 4 | stage18_n8_campaign_remainder_L8_reupload_true_s4_20260718-204645 | 2.17e-04 | 0.0017 | 0.999980 |

reupload=true's total_var change (~1.5e-4 to 2.2e-4) was never predicted
to be machine-precision -- only reupload=false carries that claim (matching
B2's structure, where reupload=true also showed a non-trivial, non-zero
total_var drift). fiber_fraction exceeds n=6's 0.99963 (HN2) in 5/5 seeds.

### reupload=false (5 seeds) — headline invariance claim

| seed | run_id | total_var max rel. change | fiber_fraction |
|---|---|---|---|
| 0 | stage18_n8_reupload_false_L8_reupload_false_s0_20260719-004647 | 4.75e-15 | 0.999667 |
| 1 | stage18_n8_reupload_false_L8_reupload_false_s1_20260719-014737 | 5.53e-15 | 0.999952 |
| 2 | stage18_n8_reupload_false_L8_reupload_false_s2_20260719-022738 | 5.77e-15 | 0.999888 |
| 3 | stage18_n8_reupload_false_L8_reupload_false_s3_20260719-030222 | 6.80e-15 | 0.999974 |
| 4 | stage18_n8_reupload_false_L8_reupload_false_s4_20260719-033543 | 7.91e-15 | 0.999756 |

Machine-precision invariance holds 5/5 (all <1e-6, same ~1e-15 order of
magnitude as HR2/HN1/HR6/HN5/B2's results), and fiber_fraction exceeds
n=6's 0.99963 in 5/5 seeds here too. **CONFIRMED** per the preregistered
rule (fiber_fraction exceeds n=6 in 10/10 runs across both arms;
reupload=false invariance holds 5/5, well past the >=4/5 bar). Both the
rotation-invariance mechanism (T5) and the fiber-fraction trend replicate
at a third qubit count.

## B4 — noisy simulation, inference-only (n=4 L=8 R0 reupload=true anchor, same checkpoint as Stage 16/16b)

Restructured per PREDICTIONS.md as an exact (shots=None) `default.mixed`
density-matrix computation over the same 180 samples, no training. Feasibility
probe: 10.9s for 180 samples at p=0.001 — no trimming needed, stretch-goal
noisy-training arm not pursued (not required for Stage 18 completion).

| p | lambda_sim(p) | equiangle_dev_z | rel. change vs exact (0.0969) |
|---|---|---|---|
| 0.001 | 0.8874 | 0.0999 | 3.1% |
| 0.005 | 0.5500 | 0.1121 | 15.7% |

lambda_sim monotonically decreasing in p (0.8874 > 0.5500), both <1 --
uniform contraction confirmed, the simulator-side bridge to HW1's hardware
contraction (lambda=0.7033) is qualitatively the same phenomenon. Angle
preservation: p=0.001 passes the <10% threshold (3.1%), but p=0.005 does
not (15.7%, inside the preregistered PARTIAL band [10%, 25%)).
**PARTIAL** per the preregistered rule -- monotonicity holds, but the
angle-preservation bullet only holds at the lower noise level. Read
literally: class-mean angular geometry survives mild depolarizing noise
but visibly distorts by p=0.005, an exact-simulation counterpart to
HW1/HW2's hardware-noise findings rather than a clean confirmation at
every tested noise level.

## B5 — hardware second backend: ibm_kingston (n=4 L=8 R0 reupload=true anchor, same checkpoint/samples as Stage 16/16b)

Same protocol as Stage 16/16b: one batched SamplerV2 job, 180 circuits,
2048 shots/circuit, same trained anchor. Job `d9egj2hhtsac739e7ts0`,
102s QPU charge time (vs the two prior ibm_marrakesh jobs' 104s each --
consistent, sanity check holds). Budget after this job: ~290s (4.8 min)
remaining, matching the preregistered accounting; this was the last
planned hardware spend for the project.

| quantity | value |
|---|---|
| lambda_kingston (HW1 fit) | 0.6709 |
| R^2 of the linear fit | 0.9356 |
| equiangle_dev_z, simulator (exact) | 0.09691 |
| equiangle_dev_z, hardware bootstrap 95% CI (1000 resamples) | [0.0813, 0.1167] |

lambda_kingston=0.6709 falls inside the preregistered [0.55, 0.85] band
(and close to HW1's ibm_marrakesh fit, lambda=0.7033 -- the contraction is
backend-consistent). The simulator's equiangle_dev_z (0.09691) falls
inside the hardware bootstrap 95% CI, so angle-survival holds by the same
methodology as HW2. **CONFIRMED** per the preregistered rule (both
bullets hold).

**Anomaly, reported factually per CLAUDE.md (no softening):** `fit_hw1`'s
secondary nc1_z-prediction step (Sigma_B pseudo-inverse extrapolation, not
part of B5's own scoring criteria) produced an unstable value
(nc1_z_predicted=146.4 vs observed 0.562, 259x relative error) on this
job, driving that function's own internal HW1-style verdict field to
REFUTED. This is a known instability of that formula when the residual's
Sigma_B matrix is near-singular -- it does not affect the lambda fit
itself (R^2=0.9356, a solid linear fit) or the angle-survival bootstrap,
which are the only two quantities B5's preregistration actually scores.
Recorded here rather than discarded, since the anomaly is real and may be
worth revisiting if HW1's prediction formula is reused on a third backend.
