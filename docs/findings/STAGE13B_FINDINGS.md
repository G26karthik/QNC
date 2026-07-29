# STAGE13B_FINDINGS.md -- Loss-visible subspace decomposition + fixed-depth rerun

Implements PLAYBOOK3.md Stage 13B / PREDICTIONS.md's Stage 13B section
(H1-H3), scored strictly against that preregistration (committed before any
Stage 13B metric below was computed, commit 8fcac2f). Corrects the original
Stage 13 matched-M sweep's interpretation: its parameter-count-matching
scheme gave R0/R1/R2/R3 unequal circuit depths (L=8/5/3/6), a depth confound
entangled with the intended measurement-expressivity axis.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS -- classical
NC definition. Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 -- quantum NC at
the optimum. San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 -- empirical
QNC study.

## Task A -- checkpoint-only recompute (matched-M sweep, no new training)

Recomputed from `results/sweeps/stage13_expressivity_20260714-111146/manifest.json`
final checkpoints (raw records: `stage13b_taskA_records.json` in that
directory). No training, no new checkpoints (CLAUDE.md invariant 3/4).

### Parameter accounting (quantifying the depth confound)

| arm | n_layers | circuit params | head/measurement params | total M | nearest Stage9 depths (fiber_fraction) |
|---|---|---|---|---|---|
| R0 | 8 | 96 | 0 | 96 | L8=0.9934 |
| R1 | 5 | 60 | 30 | 90 | L4=0.9927 / L8=0.9934 |
| R2 | 3 | 36 | 57 | 93 | L2=0.9963 / L4=0.9927 |
| R3 | 6 | 72 | 24 | 96 | L4=0.9927 / L8=0.9934 |

R1/R2's matched-M circuits (L=5, L=3) sit at or below Stage 9's shallowest
probed depths. Note the Stage 9 reference fiber_fraction values (0.99-0.996,
R0-style fixed-Z_c readout, k=3 throughout) barely move across L=2..32 --
depth alone, in the R0 sense, does not explain R1/R2's much larger drop
(0.9666, 0.9285, see STAGE12_FINDINGS.md-style table below). The drop is
better explained by k growing (9, 18) than by depth per se -- consistent
with the isotropic-null match already observed (fiber_fraction sits on
1-k/(4^n-1) for R1/R2, see PREDICTIONS.md's Stage 13B motivation). Task B
tests whether fixing depth at L=8 removes the residual R0-vs-R1/R2 gap in
the full-space witnesses that are NOT explained by k alone.

### H1 -- loss-visible subspace concentration (matched-M checkpoints)

| set | arm | wrow_per_dim | complement_per_dim | meas_per_dim | concentration (meas/wrow) | isotropic-level (complement/meas) |
|---|---|---|---|---|---|---|
| matched-M | R1 | 0.0023056 +/- 0.0001844 | 0.0037143 +/- 0.0003190 | 0.0032447 +/- 0.0002071 | 1.417 +/- 0.1514 | 1.143 +/- 0.03552 |
| matched-M | R2 | 0.0023862 +/- 0.0003560 | 0.0035194 +/- 0.0002809 | 0.0033305 +/- 0.0002493 | 1.420 +/- 0.1899 | 1.056 +/- 0.02024 |

Scoring against PREDICTIONS.md's Stage 13B H1 (preregistered thresholds:
concentration >= 2 AND isotropic-level in [0.8, 1.2], both arms):
- Concentration sub-prediction: R1 1.417, R2 1.420 -- both **below** the 2x
  threshold. Directionally consistent (head-row directions ARE more
  collapsed than the measured-span average) but not at the preregistered
  magnitude. NOT MET for either arm.
- Isotropic-level sub-prediction: R1 1.143, R2 1.056 -- both **within**
  [0.8, 1.2]. MET for both arms.

**H1 = PARTIAL** (one of two sub-predictions holds across both arms, per
the preregistered PARTIAL condition). The complement genuinely sits at the
isotropic rate (no depletion, no extra concentration); the W-row image is
modestly (not decisively) more collapsed than the span average. The
"collapse concentrates where the loss can see" story is directionally
supported but weaker than hypothesized at matched-M depths.

### H3 -- logit_nc1_z invariance (matched-M set)

| arm | logit_nc1_z |
|---|---|
| R0 | 0.279491 +/- 0.0658346 |
| R1 | 0.282940 +/- 0.0450247 |
| R2 | 0.229646 +/- 0.0656241 |
| R3 | 0.228335 +/- 0.0311263 |

Mann-Whitney U, R0 vs each other arm:

| arm | logit_nc1_z p-value |
|---|---|
| R1 | 0.8413 |
| R2 | 0.5476 |
| R3 | 0.2222 |

All three pairs p >= 0.05 in the matched-M set: no significant difference
from R0's logit_nc1_z. Matched-M half of H3's scoring condition MET; full
H3 scoring requires the fixed-depth comparison too (Task B, below).

### Figure

`results/sweeps/stage13_expressivity_aggregate/expressivity.png` regenerated
with the isotropic dimension-counting null (1 - k/(4^n-1)) added to the
fiber_fraction panel (dashed grey line, per-arm k).

## Task B -- fixed-depth rerun (R1/R2 at L=8, 10 new runs)

Sweep config `configs/sweeps/stage13b_fixed_depth.yaml`: R1 and R2 rerun with
circuit fixed at L=8 (96 circuit params, matching R0), heads unchanged (30
and 57 params respectively, reported separately per PREDICTIONS.md, not
padded into M). 5 seeds each, blobs3 C=3 n=4, 600 epochs. R0/R3 reused
unchanged from the matched-M sweep (already L=8/L=6).

The background sweep process was killed by the environment partway through
(R1 all 5 seeds and R2 seeds 0-2 completed; R2 seed 3 died mid-training at
epoch 460/600, left in place uncleaned per CLAUDE.md invariant; seed 4 never
started). The missing two R2 seeds were completed with a small recovery
sweep, `configs/sweeps/stage13b_fixed_depth_r2_remainder.yaml`. The full
10-run manifest was assembled by hand from the three partial run logs:
`results/sweeps/stage13b_fixed_depth_20260714-143839/manifest.json`. Raw
recomputed records: `stage13b_taskB_records.json` in that directory.

### H2 -- fixing depth removes the full-space gap

| arm | qnc1_fisher | purity_mean | overlap_offdiag | fiber_excess_above_null |
|---|---|---|---|---|
| R0 (matched-M, unchanged) | 29.7905 +/- 4.09096 | 0.105038 +/- 0.00637682 | 0.0591396 +/- 0.000482845 | 0.00518 +/- 0.000532 |
| R1 (fixed L=8) | 29.1499 +/- 2.45564 | 0.104559 +/- 0.00365032 | 0.0581849 +/- 0.00094441 | 0.00281 +/- 0.000990 |
| R2 (fixed L=8) | 27.659 +/- 2.90339 | 0.107583 +/- 0.00449335 | 0.0586461 +/- 0.00128154 | 0.00432 +/- 0.001839 |

Mann-Whitney U, R0 vs each arm:

| arm | qnc1_fisher | purity_mean | overlap_offdiag | fiber_excess_above_null |
|---|---|---|---|---|
| R1 | 0.5476 | 0.6905 | 0.1508 | **0.007937** |
| R2 | 0.4206 | 0.4206 | 0.8413 | 0.6905 |

All three full-space witnesses (qnc1_fisher, purity_mean, overlap_offdiag):
p >= 0.05 for BOTH fixed-depth R1 and R2 vs R0 -- the matched-M sweep's
apparent R0 > R1 > R2 full-space weakening (29.8 -> 19.5 -> 12.8 qnc1_fisher)
collapses to statistical noise once depth is fixed (29.8 -> 29.1 -> 27.7).
This is a clean depth-confound correction.

fiber_excess_above_null does NOT fully converge for R1: p=0.0079 (still
significant, though the matched-M gap shrank -- R1 excess 0.0019 matched-M
vs 0.0028 fixed-depth, closer to R0's 0.0052 but not statistically
indistinguishable). R2 converges cleanly (p=0.69).

**H2 = PARTIAL** (full-space witnesses match R0 within noise for both arms
-- the depth-confound explanation is strongly supported there; the
fiber-excess-above-null sub-condition holds for R2 but not R1, a strict
subset of the measured quantities, per the preregistered PARTIAL condition).

### H3 -- logit_nc1_z invariance (fixed-depth set)

| arm | logit_nc1_z |
|---|---|
| R0 (matched-M) | 0.279491 +/- 0.0658346 |
| R1 (fixed L=8) | 0.233615 +/- 0.0416885 |
| R2 (fixed L=8) | 0.186064 +/- 0.0122125 |

Mann-Whitney U, R0 vs each arm: R1 p=0.3095, R2 p=0.1508 -- both >= 0.05.

Combined with the matched-M comparison (R1 p=0.8413, R2 p=0.5476, R3
p=0.2222, all >= 0.05): **H3 = CONFIRMED** across both the matched-M and
fixed-depth comparisons, all arm pairs. logit_nc1_z (NC1 on the actual
CE-loss input) is statistically indistinguishable from R0's regardless of
readout family or circuit depth in this sweep.

### H1 robustness check -- subspace split on fixed-depth checkpoints

| set | arm | wrow_per_dim | complement_per_dim | meas_per_dim | concentration (meas/wrow) | isotropic-level (complement/meas) |
|---|---|---|---|---|---|---|
| matched-M | R1 | 0.0023056 | 0.0037143 | 0.0032447 | 1.417 +/- 0.1514 | 1.143 +/- 0.03552 |
| matched-M | R2 | 0.0023862 | 0.0035194 | 0.0033305 | 1.420 +/- 0.1899 | 1.056 +/- 0.02024 |
| fixed-depth | R1 | 0.0022271 | 0.0037356 | 0.0032328 | 1.464 +/- 0.1322 | 1.156 +/- 0.03067 |
| fixed-depth | R2 | 0.0019763 | 0.0035483 | 0.0032863 | 1.676 +/- 0.1511 | 1.080 +/- 0.01124 |

The pattern reproduces at fixed depth: concentration factors (1.46, 1.68)
still fall short of the preregistered 2x threshold; isotropic-level factors
(1.16, 1.08) stay within [0.8, 1.2]. H1's PARTIAL scoring is depth-robust --
it is not an artifact of the matched-M sweep's shallower R1/R2 circuits.

## Interpretation

| hypothesis | verdict |
|---|---|
| H1 | PARTIAL (both matched-M and fixed-depth checkpoints) |
| H2 | PARTIAL |
| H3 | CONFIRMED |

**Stage 13 flag (per Task C):** the original Stage 13 matched-M table and
`results/sweeps/stage13_expressivity_20260714-111146/expressivity_table.md`
figure/data are kept unmodified, but their interpretation is corrected: the
apparent R0 > R1 > R2 full-space-witness weakening was substantially a
circuit-depth artifact of the parameter-matching scheme (R1 L=5, R2 L=3 vs
R0/R3's L=8/L=6), not a pure measurement-expressivity effect. Fixing depth
at L=8 removes that gap for all three full-space witnesses. The
fiber_fraction gap, by contrast, is real but is largely explained by the
isotropic dimension-counting null 1-k/(4^n-1) (k=9 for R1, k=18 for R2) --
not by depth, and not by a strong measurement-subspace collapse signal: the
H1 decomposition shows the loss-visible (head-row) directions are only
modestly (~1.4-1.7x, not the preregistered >=2x) more collapsed than the
measured-span average, while the in-span complement sits close to the
isotropic rate. logit_nc1_z (H3) is invariant throughout, meaning whatever
collapse pressure exists reaches the actual CE-loss input equally across all
four readout families and both depths tested.
