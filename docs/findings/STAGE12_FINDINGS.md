# STAGE12_FINDINGS.md -- Preregistered reanalysis: measurement-subspace collapse

Implements PLAYBOOK3.md Stage 12 / SPEC3_ADDENDUM.md sections 17-18. Scored
strictly against PREDICTIONS.md (committed before any metric below was
computed). No new training runs -- every number is recomputed from existing
checkpoints via seeded reconstruction (epoch 0) and state_dict reload.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

## P1 -- NC1_z / r_z decay post-TPT (CONFIRMED)

| arm | metric | epoch-0 | final | final/init |
|---|---|---|---|---|
| L=8 anchor | NC1_z | 62.06 +/- 41.17 | 0.2795 +/- 0.06583 | 0.004504 |
| L=8 anchor | r_z | 43 +/- 18.59 | 0.8908 +/- 0.1638 | 0.02072 |
| C=2 anchor | NC1_z | 18.06 +/- 11.98 | 0.1279 +/- 0.01855 | 0.00708 |
| C=2 anchor | r_z | 68.05 +/- 42.71 | 0.7575 +/- 0.138 | 0.01113 |

Threshold: final/init <= 0.1 (>= 1 order of magnitude decay) on all 4 rows
for CONFIRMED; 0 rows for REFUTED; otherwise PARTIAL.

## P2 -- beta * mean_margin grows post-TPT (CONFIRMED)

| arm | epoch-0 | final |
|---|---|---|
| L=8 anchor | -0.8675 | 8.545 |
| C=2 anchor | 0.03529 | 8.424 |

## P3 -- box vertices vs simplex ETF, C=3, L=8 anchor final epoch (REFUTED)

d_boxn (normalized d_box) = 0.7122 +/- 0.01767
d_etf (equiangle_dev_z)    = 0.07222 +/- 0.02453

**Correction note (Stage 17b, appended, original REFUTED verdict above
unchanged):** the d_boxn/d_etf comparison correctly refutes the box-vertex
model on MAGNITUDE (d_boxn is an order of magnitude worse than d_etf), but
d_etf alone cannot refute it on ANGLE, because at C=3 the centered box
vertices themselves have pairwise cosine exactly -1/(C-1) = -1/2 -- the
identical angular target as the simplex ETF (proved algebraically,
`test_box_vertices_c3_cosine_equals_etf_target`, `tests/test_metrics.py`).
The sharper, corrected statement (THEORY_NOTES.md T2'): class means adopt
simplex-ETF DIRECTIONS (shared with the box-vertex model at C=3) without
ever approaching the box-vertex model's fixed saturation RADIUS
(sqrt(8/3) ~= 1.6330); the L=8 anchor's final-epoch class-mean radius
(recomputed this pass, existing checkpoints, no new training) is
0.3287 +/- 0.0257, ~20.1% of that saturation value, up from 0.0649 +/-
0.0164 at epoch 0. The P3 REFUTED verdict stands (box vertices are refuted,
and correctly so on magnitude grounds); this note only sharpens WHY, since
the angle-only part of the original argument was weaker than stated.

## P4 -- fiber decomposition dichotomy, L=8 anchor (PARTIAL)

- V_meas: epoch-0 0.01038 -> final 0.005896 (decays: True)
- V_fiber: final 0.8892 +/- 0.006032 vs untrained-null
  0.9447 +/- 0.007092 (flat within 2 std: False)
- fiber_fraction: final 0.9934 (>= 0.9: True)

## P5 -- untrained null shows no z-space collapse (CONFIRMED)

Mann-Whitney U, L=8-anchor final NC1_z (n=5, mean 0.2795)
vs untrained-null NC1_z (n=10, mean 19.93): p = 0.000666

## Stage 9 L-sweep, full-depth context (final epoch, all 8 depths)

| L | NC1_z (final) | fiber_fraction (final) |
|---|---|---|
| L2 | 0.8621 +/- 0.2079 | 0.9963 +/- 0.002025 |
| L4 | 0.4872 +/- 0.07124 | 0.9927 +/- 0.0004155 |
| L8 | 0.2795 +/- 0.06583 | 0.9934 +/- 0.0005321 |
| L12 | 0.1798 +/- 0.02936 | 0.9933 +/- 0.0003375 |
| L16 | 0.09233 +/- 0.005544 | 0.9949 +/- 0.0004883 |
| L21 | 0.05846 +/- 0.006548 | 0.9954 +/- 0.0002915 |
| L26 | 0.04116 +/- 0.005724 | 0.9951 +/- 0.0001507 |
| L32 | 0.02601 +/- 0.005393 | 0.9959 +/- 0.0004426 |

## Interpretation

Scored outcome per PLAYBOOK3.md Stage 12 gate: P1=CONFIRMED, P2=CONFIRMED,
P3=REFUTED, P4=PARTIAL, P5=CONFIRMED. Reported factually; no
softening of REFUTED/PARTIAL results (CLAUDE.md).

## Supplements (PLAYBOOK3.md Stage 13 Task 1, existing checkpoints only)

Three diagnostic figures added to the L=8 anchor (5 seeds) reanalysis, no new
training or checkpoints:

- `results/transition_L_L8_aggregate/figures/etf_formation.png`: d_etf
  (equiangle_dev_z) and pairwise z-space cosines vs epoch, E0 marked per seed
  -- shows when ETF geometry forms relative to TPT onset.
- `results/transition_L_L8_aggregate/figures/nc1z_decomposition.png`:
  tr(Sigma_W^z) and tr(Sigma_B^z) vs epoch separately (within-class shrink vs
  between-class growth), same runs.
- `results/transition_L_L8_aggregate/figures/fiber_dip_diagnostic.png`:
  V_fiber vs epoch against the untrained-null band, showing when the ~6% dip
  (P4 supplement) develops.
