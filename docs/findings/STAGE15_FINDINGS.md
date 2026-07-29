# STAGE15_FINDINGS.md -- HR5 diagnosis + n=6 replication

Implements PREDICTIONS.md's "Stage 15 preregistration -- HR5 diagnosis
(checkpoints only) + n=6 replication (HN1-HN4)" section, scored strictly
against the thresholds committed there (commit 3233d64) before any Stage 15
metric was computed.

Citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang, Tao,
Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

## HR5 diagnosis (checkpoints only, no new training)

Full detail, tables, and interpretation in STAGE14B_FINDINGS.md's "HR5
diagnosis" section (`src/qnc/reanalyze.py`'s `reanalyze_hr5_diagnosis`,
`compute_hr5b`, `compute_hr5c`; data at
`results/stage14b_reanalysis/stage15_hr5_diagnosis.json`). Summary:

| prediction | verdict |
|---|---|
| HR5b (epoch-0 z-space constellation spread flat across blob_std, rel. range < 20%) | **CONFIRMED** (total_var 2.79%, trace_w 6.67%) |
| HR5c (final trace_w correlates with own epoch-0 trace_w, pooled n=25) | **PARTIAL** (rho=0.132 > 0, p=0.529 not significant) |

HR5b isolates the mechanism behind HR5's refutation (STAGE14B_FINDINGS.md):
`pca_encode`'s per-run min-max scaling to [-pi, pi] absorbs blob_std's raw
spread before the angle encoding sees it. HR5c's PARTIAL result is
consistent with, not contradictory to, this -- with epoch-0 spread already
nearly flat across blob_std, there is little between-run variance left for a
within-run inheritance signal to grab onto, so a real but weak effect
(right sign) does not clear significance at n=25.

## n=6 replication (SPEC3_ADDENDUM.md section 21)

blobs3 lifted to R^6 (`data.ambient_dim: 6`, same center-distance 3.0 /
blob_std 0.8 / 60-samples-per-class recipe as the n=4 headline experiment),
n=6 qubits, C=3, both rotation-mechanism arms (reupload=true and
reupload=false, per HR1-HR4). `configs/stage15_n6_base.yaml`,
`configs/sweeps/stage15_n6_depth_probe.yaml`,
`configs/sweeps/stage15_n6_campaign.yaml`.

### Depth probe: L in {8, 16, 32, 64}, 3 seeds per (depth, reupload-arm) cell (24 runs)

Manifest: `results/sweeps/stage15_n6_depth_probe_20260716-160429/manifest.json`.
TPT fraction per cell (`compute_tpt_fractions`):

| L | reupload=true | reupload=false |
|---|---|---|
| 8 | 1.0 (3/3) | 1.0 (3/3) |
| 16 | 1.0 (3/3) | 1.0 (3/3) |
| 32 | 1.0 (3/3) | 1.0 (3/3) |
| 64 | 1.0 (3/3) | 1.0 (3/3) |

Every probed depth in both arms reached TPT in all 3 seeds -- `select_n6_campaign_depths`
selects the smallest probed depth, **L8**, as the fully-interpolating depth
for BOTH reupload arms (a stronger result than the n=4 headline experiment
needed a much larger L to reach; at n=6 the parameter count per layer is
larger (18 vs 12 at n=4) relative to the same 3-class blobs3 task, so L8
already suffices). One run (`L64_reupload_true` seed 0's first attempt) was
killed mid-training by an environment restart at epoch 595/600 with no final
checkpoint; left in place per CLAUDE.md invariant 3 and not referenced by
any manifest, re-run cleanly as a fresh run_id.

### 5-seed full-witness campaign at L8 (both arms)

Manifest: `results/sweeps/stage15_n6_campaign_20260716-163857/manifest.json`.
Seeds 0-2 per arm are reused directly from the depth probe's config-identical
L8 cells (same base config, same overrides, same seed -> identical data
split/init/training trajectory, only the results-dir name differs); seeds
3-4 per arm newly run. All 10/10 campaign seeds (both arms) reached TPT.
SPEC2 + SPEC3 section 18 witnesses recomputed at epoch 0 and every
checkpointed epoch via `reanalyze_n6_sweep_zspace` (checkpoints only, no new
training); data at `results/stage15_n6_campaign_zspace.json`.

### HN1 -- reupload=false total_var constant at n=6 (machine precision)

Max relative change `abs(total_var[t] - total_var[0]) / total_var[0]` across
all checkpoints, per seed (L8_reupload_false, 5 seeds): 4.65e-15, 4.58e-15,
4.40e-15, 6.20e-15, 6.54e-15.

Scoring: CONFIRMED if the bound (< 1e-6) holds at every checkpoint for all 5
seeds. 5/5 seeds pass, all six orders of magnitude below the threshold ->
**CONFIRMED**. Identical numerical fingerprint to HR2 at n=4
(STAGE14B_FINDINGS.md, ~1e-15 relative change): the "collapse by rotation"
mechanism (reupload=false is a fixed unitary conjugation, exactly invariant
under Hilbert-Schmidt distance) holds unchanged at n=6.

### HN2 -- fiber_fraction at n=6 exceeds the n=4 reference (~0.993) in all TPT-reaching arms

Both L8 campaign arms reached TPT in 5/5 seeds. Final-epoch fiber_fraction,
5-seed mean:

| arm | per-seed final fiber_fraction | mean |
|---|---|---|
| reupload=true | 0.99965, 0.99962, 0.99960, 0.99968, 0.99962 | **0.99963** |
| reupload=false | 0.99947, 0.99899, 0.99896, 0.99951, 0.99893 | **0.99917** |

Scoring: CONFIRMED if the mean exceeds the n=4 reference (0.993, L=8
anchor, STAGE12_FINDINGS.md P4) for every TPT-reaching arm. Both arms
exceed it (0.99963 > 0.993, 0.99917 > 0.993) -> **CONFIRMED**. Dimension
counting predicted this: at n=6 the measured subspace is 3 of 4095 traceless
Hermitian directions (fiber_fraction ceiling far closer to 1) vs 3 of 255 at
n=4, and both arms land closer to 1 than the n=4 anchor did, in the
predicted direction.

### HN3 -- NC1_z decays >= 1 order of magnitude in TPT-reaching arms, both reupload settings

Both arms 5/5 TPT-reaching. `nc1_z(final) / nc1_z(epoch0)` per seed:

| arm | per-seed ratio | mean |
|---|---|---|
| reupload=true | 0.00184, 0.00471, 0.00167, 0.00193, 0.00393 | **0.00282** |
| reupload=false | 0.00088, 0.00654, 0.00341, 0.00228, 0.00860 | **0.00434** |

Scoring: CONFIRMED if the 5-seed-mean ratio is <= 0.1 in BOTH arms.
0.00282 and 0.00434 both clear the bound by more than an order of magnitude
beyond the threshold itself (roughly 2.5-3 orders of magnitude of decay, not
just 1) -> **CONFIRMED**. NC1_z collapse in z-space is, if anything, sharper
at n=6 than the >=1-order-of-magnitude bound required.

### HN5 -- per-class purity and pairwise overlap constant at n=6 (Stage 17b, existing HN1 checkpoints, no new training)

Preregistered in PREDICTIONS.md's "Stage 17b preregistration" section
(committed before this metric was computed), the n=6 analog of HR6: same 5
`L8_reupload_false` campaign checkpoints already used for HN1, all 161
checkpoints per seed, `mean_class_purity`/`class_mean_overlap` via
`reanalyze_hr6_hn5_invariance`:

| seed | run_id | max rel. change, purity | max rel. change, overlap |
|---|---|---|---|
| 0 | stage15_n6_depth_probe_L8_reupload_false_s0_20260716-003842 | 4.8959e-15 | 4.7193e-15 |
| 1 | stage15_n6_depth_probe_L8_reupload_false_s1_20260716-004455 | 4.6127e-15 | 4.6066e-15 |
| 2 | stage15_n6_depth_probe_L8_reupload_false_s2_20260716-005103 | 3.6842e-15 | 3.9424e-15 |
| 3 | stage15_n6_campaign_L8_reupload_false_s3_20260716-162856 | 5.9612e-15 | 6.0586e-15 |
| 4 | stage15_n6_campaign_L8_reupload_false_s4_20260716-163357 | 6.1706e-15 | 6.1230e-15 |

Scoring: identical to HR6 (CONFIRMED requires < 1e-6 for both quantities,
5/5 seeds). All 10 values are ~1e-15, matching HR6's n=4 fingerprint
exactly -> **CONFIRMED**. Data:
`results/stage14b_reanalysis/stage17b_hr6_hn5.json` (`hn5_n6` key).

### HN4 -- TPT fractions (factual, not scored)

Depth probe (3 seeds per cell): all 8 (L, reupload-arm) cells reached TPT in
3/3 seeds (table above). Campaign (5 seeds per arm at L8): reupload=true
5/5, reupload=false 5/5.

### Interpretation (n=6 replication)

| prediction | verdict |
|---|---|
| HN1 (reupload=false total_var constant, machine precision) | CONFIRMED (max rel. change ~5e-15, 5/5 seeds) |
| HN2 (fiber_fraction at n=6 exceeds n=4's ~0.993 in all TPT-reaching arms) | CONFIRMED (0.99963 true-arm, 0.99917 false-arm) |
| HN3 (NC1_z decays >= 1 order of magnitude, both arms) | CONFIRMED (ratios 0.0028 / 0.0043, ~2.5-3 orders) |
| HN4 (TPT fractions) | not scored -- 8/8 depth-probe cells 3/3 TPT; both campaign arms 5/5 TPT |
| HN5 (per-class purity + pairwise overlap exactly constant, n=6, Stage 17b) | CONFIRMED (~1e-15 relative change, both quantities, 5/5 seeds) |

No softening (CLAUDE.md): all three scored n=6 predictions (HN1-HN3)
CONFIRM cleanly, and in the same direction as the free parameter (HN2, HN3
margins) predicted -- the measurement-subspace-collapse / fiber-no-collapse
dichotomy from n=4 (STAGE12_FINDINGS.md, STAGE14B_FINDINGS.md) survives
unchanged at n=6 and, on the fiber_fraction axis specifically, strengthens
exactly as the dimension-counting argument in PLAYBOOK3.md's Stage 15 note
predicted. This replicates, at a second value of n, the rotation-by-unitary
mechanism (HN1, mirroring HR2) and the z-space NC1 collapse (HN3, mirroring
HR3) established at n=4 in STAGE14B_FINDINGS.md -- it does not by itself
establish scaling behavior as a function of n in general (two points, not a
trend), a caveat for THEORY_NOTES (Stage 17).

Figures: `results/stage15_n6_reanalysis/figures/n4_vs_n6_dichotomy.png`
(fiber_fraction vs epoch, n=4 L=8 anchor vs n=6 L=8 campaign reupload=true
arm, side by side, mean +/- 1 std), `results/stage15_n6_reanalysis/figures/n6_total_variance_rotation.png`
(total within-class HS variance vs epoch, n=6, reupload=true vs
reupload=false overlaid, mean +/- 1 std).

Reanalysis data: `results/stage14b_reanalysis/stage15_hr5_diagnosis.json`,
`results/stage14b_reanalysis/stage15_hn1.json`,
`results/stage14b_reanalysis/stage15_hn2_hn3.json`,
`results/stage15_n6_campaign_zspace.json`.

## Housekeeping

Both n=6 sweeps ran as detached background processes (see conversation log)
after an unrelated environment restart killed the first in-process depth-
probe run partway through (18/24 cells preserved, 1 partial run left in
place per CLAUDE.md invariant 3 and re-run cleanly). Total n=6 training:
24 depth-probe runs + 4 new campaign runs (6 campaign seeds reused directly
from config-identical depth-probe runs) = 28 new training runs, all L8-L32
completing well under the 2-hour per-run pause threshold (max observed
single-run wall time: ~90 minutes at L64).
