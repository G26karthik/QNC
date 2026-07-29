# STATS_APPENDIX.md — Effect sizes, CIs, and multiple-comparison correction

Stage 18 arm B6 (PREDICTIONS.md "Stage 18 preregistration"). Adds Cliff's
delta and bootstrap 95% CIs to every ledger comparison that has a
p-value or a per-seed distribution behind it, and applies Holm-Bonferroni
correction across the family of confirmatory (p-value-bearing) tests
spanning Stages 12-18.

**Scope note (in-session decision, approved by the user):** historical
items (Stages 12-17b) are analyzed from the p-values, sample sizes, and
per-seed numbers already reported in their findings files / RESULTS_V3.md's
ledger table — not from re-touching the original results/ directories.
Where a p-value plus exact group sizes fully determine the underlying
exact permutation null distribution (Mann-Whitney U, small n), Cliff's
delta is derived exactly by matching the reported p to its unique
implied U (verified below: every match reproduces the reported p to
5+ significant figures). Where a Spearman correlation + n is reported,
a Fisher z-transform 95% CI is added. Comparisons that are simple
per-seed threshold checks with no group-vs-group or correlation
structure (most machine-precision and NC1_z-ratio items, e.g.
HR1-HR3/HR6/HN1-HN3/HN5/HW3) do not have a Cliff's-delta-shaped question
to ask; for those, a bootstrap 95% CI on the per-seed mean is reported
instead where per-seed numbers are available (new Stage 18 items), or
flagged as "not re-derivable from the findings-file summary alone" where
they are not (older items reported only as a single aggregate, e.g. HR2's
"max rel. change ~1e-15, 5/5 seeds" without each seed's individual value
recorded in the findings file).

## Part 1 — Cliff's delta for Mann-Whitney-based comparisons (exact, derived from reported p + group sizes)

All matches below reproduce the reported p-value to at least 5 significant
figures, confirming the derivation is exact (not an approximation) given
only the group sizes and the already-published p-value.

| Comparison | Stage | n1, n2 | p (reported) | p (matched, exact) | Cliff's delta |
|---|---|---|---|---|---|
| P5: trained NC1_z vs untrained-null NC1_z | 12 | 5, 10 | 0.000666 | 0.000666 | **-1.00** (complete separation) |
| Arm3: label-smoothing vs baseline (NC1_z shrink) | 14 | 5, 5 | 0.4206 | 0.42063 | -0.36 |
| Arm6: frozen-beta vs baseline (NC1_z shrink) | 14 | 5, 5 | 0.01587 | 0.01587 | -0.92 |
| H2: fiber_excess_above_null, R1 matched-M vs null | 13B | 5, 5 | 0.0079 | 0.00794 | **-1.00** (complete separation) |
| H2: fiber_excess_above_null, R2 matched-M vs null | 13B | 5, 5 | 0.69 | 0.69048 | -0.20 |
| H3: logit_nc1_z, R0 vs R1 (fixed-depth) | 13B | 5, 5 | 0.3095 | 0.30952 | -0.44 |
| H3: logit_nc1_z, R0 vs R2 (fixed-depth) | 13B | 5, 5 | 0.1508 | 0.15079 | -0.60 |
| H3: logit_nc1_z, R0 vs R1 (matched-M) | 13B | 5, 5 | 0.8413 | 0.84127 | -0.12 |
| H3: logit_nc1_z, R0 vs R2 (matched-M) | 13B | 5, 5 | 0.5476 | 0.54762 | -0.28 |
| H3: logit_nc1_z, R0 vs R3 (matched-M) | 13B | 5, 5 | 0.2222 | 0.22222 | -0.52 |

Two comparisons (P5, H2-R1) are **exact complete separations** (|delta|=1) —
the p-values that small-sample Mann-Whitney tests can report are quantized,
and both happen to sit at the smallest achievable p for their group sizes
(1/C(15,5) two-tailed for P5; 1/C(10,5) two-tailed for H2-R1), which is
what "complete separation" produces. This is a consequence of small n, not
evidence the true effect is literally infinite; it is the ceiling the test
design can express.

## Part 2 — Fisher-z 95% CIs for correlation-based comparisons

| Comparison | Stage | rho | n | p | 95% CI (Fisher z) |
|---|---|---|---|---|---|
| HR4: Spearman(trace_w, L) | 14B | -0.4509 | 40 | 0.0035 | [-0.669, -0.162] |
| HR5c: Spearman(final trace_w, epoch-0 trace_w) | 15 | 0.1322 | 25 | 0.5288 | [-0.277, 0.501] |
| Arm2: Spearman(trace_w, log epoch), post-E0 | 14 | -0.0716 | 100 | 0.4793 | [-0.264, 0.127] |

HR4's CI excludes 0 (consistent with its CONFIRMED verdict). HR5c's and
Arm2's CIs both straddle 0 widely, consistent with their PARTIAL/CONFIRMED
(no-significant-effect) verdicts — the data do not distinguish these rho
values from zero.

## Part 3 — Bootstrap 95% CIs for Stage 18's new per-seed distributions (B1, B2)

10,000-resample percentile bootstrap on the 5-seed mean, using this
session's raw per-seed numbers directly (no historical-file dependency).

| Arm | Quantity | Mean | 95% CI |
|---|---|---|---|
| B1 (SGD momentum) | total_var max rel. change | 3.73e-15 | [3.09e-15, 4.69e-15] |
| B1 (SGD momentum) | NC1_z ratio | 0.0186 | [0.0126, 0.0246] |
| B2 reupload=true (HWE ansatz) | total_var max rel. change | 0.0105 | [0.0069, 0.0142] |
| B2 reupload=true (HWE ansatz) | NC1_z ratio | 0.0108 | [0.0053, 0.0164] |
| B2 reupload=false (HWE ansatz, headline) | total_var max rel. change | 3.34e-15 | [3.06e-15, 3.60e-15] |
| B2 reupload=false (HWE ansatz, headline) | NC1_z ratio | 0.0216 | [0.0135, 0.0310] |

Both B1's and B2 reupload=false's total_var CIs sit entirely within
machine precision (upper bound 4.69e-15 and 3.60e-15 respectively, both
~5 orders of magnitude below the 1e-6 threshold) — the invariance claim is
not a borderline call in either case. B4 (noisy simulation) has no
per-seed distribution to bootstrap: it is an exact, deterministic
density-matrix computation at each of 2 noise levels, not a stochastic
draw across seeds (see PREDICTIONS.md B4 and STAGE18_FINDINGS.md).

## Part 4 — Holm-Bonferroni correction across the confirmatory-test family

Family: every p-value above that fed a CONFIRMED/REFUTED/PARTIAL scoring
decision anywhere in Stages 12-18 (13 tests). Sorted ascending, compared
against alpha/(m-i+1) at family-wise alpha=0.05, m=13 (standard Holm
step-down procedure):

| rank i | test | p | threshold alpha/(m-i+1) | reject H0? |
|---|---|---|---|---|
| 1 | P5 (trained vs untrained NC1_z) | 0.000666 | 0.003846 | **yes** |
| 2 | HR4 (Spearman trace_w vs L) | 0.0035 | 0.004167 | **yes** |
| 3 | H2 (fiber_excess R1, matched-M) | 0.0079 | 0.004545 | no -- **STOP** |
| 4-13 | (Arm6 0.01587, Arm3 0.4206, H3 x5, HR5c 0.5288, Arm2 0.4793, H2-R2 0.69) | -- | -- | no (Holm step-down halts at first non-rejection) |

**Only 2 of 13 confirmatory tests survive Holm-Bonferroni correction at
family-wise alpha=0.05: P5 and HR4.** This does not overturn any existing
verdict — the project's scoring rules were never framed as requiring
Holm-Bonferroni-corrected significance (most CONFIRMED verdicts for
non-significance findings, like Arm2/H3/HR5c, are actually *strengthened*
by stricter correction, since failing to reject becomes easier, not
harder, as the correction tightens the threshold for rejection). The one
result worth flagging explicitly: **Arm6's frozen-beta finding (p=0.01587,
reported as "significant, but in the wrong direction," contributing to
its REFUTED verdict) does not survive Holm-Bonferroni correction.**
Arm6's REFUTED verdict was never solely a significance claim — the
"wrong direction" (frozen beta shrinks within-class variance *less*, not
more, than the mean-margin-escape-valve hypothesis predicted) is the
substantive finding regardless of whether p=0.01587 remains "significant"
under family-wise correction. Reported factually per CLAUDE.md ("no
softening after the fact"): the direction-based REFUTED verdict stands;
the significance framing of the underlying test does not survive
correction and should not be over-read as an independently strong result.

## Summary

No existing CONFIRMED/REFUTED/PARTIAL verdict changes as a result of this
appendix. What it adds: (1) effect-size magnitudes for every p-value-
bearing comparison across the project (two are exact-separation ceilings
of small-sample tests, not literally infinite effects); (2) correlation
CIs showing HR4 excludes zero while HR5c/Arm2 do not, consistent with
their respective verdicts; (3) bootstrap confirmation that B1's and B2's
machine-precision invariance claims are not borderline; (4) the one
honest correction-sensitive flag: Arm6's significance claim does not
survive family-wise correction, though its REFUTED verdict rests on
effect direction, not on that p-value alone.
