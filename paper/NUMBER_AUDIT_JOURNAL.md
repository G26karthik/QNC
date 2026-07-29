# NUMBER_AUDIT_JOURNAL.md — 15 sampled claims traced to source

Sampled from `paper_journal/main.tex` (Springer *Quantum Machine
Intelligence* submission draft), spanning theory, the synthetic controlled
arm, real vision-data results, the comparative study, the generality
(breadth-sprint) arms, hardware validation, and the preregistration ledger.
Each row gives the paper's wording, the exact source file it was copied
from, and a match verdict. No number below was hand-recomputed or rounded
beyond what the source itself reports, except the ledger extension through
Stage 20 (row 15), which is explicitly derived in this pass and flagged as
such.

| # | Claim in paper (main.tex) | Source | Match |
|---|---|---|---|
| 1 | "maximum relative change in total variance ... on the order of $10^{-15}$ ... nine orders of magnitude below our preregistered $10^{-6}$ threshold ... NC1$_z$ still decays by a factor of $0.0209$" (§5.6) | `STAGE14B_FINDINGS.md` (HR2: 3.56e-15–5.84e-15, 5/5 seeds; HR3: ratio 0.0209, 5/5 TPT) | Exact |
| 2 | "fiber\_fraction exceeds the $n=6$ reference in 10/10 runs ... range $0.999667$–$0.999980$" ($n=8$, §5.7) | `STAGE18_FINDINGS.md` Arm B3 tables (reupload=true: 0.999975–0.999980; reupload=false: 0.999667–0.999974) | Exact |
| 3 | Table 1 (real-data TPT attempt, MNIST, $L{=}16$): spc=50 reupload=false test acc. 0.927, med. NC1$_z$ 0.02706; spc=100 reupload=false TPT 4/5, test acc. 0.940 | `STAGE20_FINDINGS.md` Arm W1 spc=50/spc=100 tables | Exact |
| 4 | "its total\_var relative change ($5.3\times10^{-15}$) is indistinguishable from the four TPT-reaching seeds" (isolation control, §6.2) | `STAGE20_FINDINGS.md` Arm W1 spc=100 table, seed 1 (`total_var_max_rel_change`=5.31e-15, TPT reached=false) | Exact |
| 5 | "test accuracy sits at chance ($0.367$, $C=3$)" (amplitude encoding, reupload=true, §6.3) | `STAGE20_FINDINGS.md` Arm W4 table (both probed seeds, test\_acc\_final=0.367) | Exact |
| 6 | "classical $\mathrm{tr}(\Sigma_W)$ ... grows $27$--$163\times$ under training" (§7.1) | `STAGE20_FINDINGS.md` Arm W2 seed-ratio table (MNIST 29.6/27.3/36.7; Fashion-MNIST 31.4/52.8/163.4; min 27.3, max 163.4) | Exact |
| 7 | Table 2/3 (MNIST/Fashion-MNIST comparative study, per-dataset split with "Abs. contraction" column: $31.2\times$/$82.5\times$ increase) | `STAGE20_FINDINGS.md` Arm W3 per-dataset split tables | Exact |
| 8 | "SGD momentum ... TPT 5/5 ... NC1$_z$ decays at least one order of magnitude in 5/5 seeds (max ratio 0.0272)" (§8) | `STAGE18_FINDINGS.md` Arm B1 5-seed campaign table (ratios 0.0139–0.0272, all TPT True) | Exact |
| 9 | "hardware-efficient ansatz ... total\_var machine-precision invariance holds 5/5 seeds (all $<10^{-6}$" (§8) | `STAGE18_FINDINGS.md` Arm B2 reupload=false table (2.81e-15–3.71e-15, 5/5) | Exact |
| 10 | "$\lambda_\mathrm{sim}(p)$ decreases monotonically ... $0.8874$ at $p{=}0.001$, $0.5500$ at $p{=}0.005$" (noisy inference, §8) | `STAGE18_FINDINGS.md` Arm B4 table | Exact |
| 11 | Table 4 (`ibm_kingston`: $\lambda=0.6709$, $R^2=0.9356$; CI $[0.0813,0.1167]$) and $259\times$ relative-error anomaly (§9.2) | `STAGE18_FINDINGS.md` Arm B5 table and "Anomaly" paragraph | Exact |
| 12 | "untrained-parameter control ... $\mathrm{NC1}_z$ $94.2\times$ the trained hardware value" (`ibm_marrakesh`, §9.1) | `RESULTS_V3.md` §(d), HW3 paragraph ("ratio 94.2x") | Exact |
| 13 | "single-input quantum Fisher information rank saturates at exactly $30=2(2^4-1)$" (§5.5) | `RESULTS_V3.md` §(e), Larocca discussion ("saturates at exactly 30 ... 2(2^4-1)=2(dim-1)") | Exact |
| 14 | Proposition 5 (ETF) empirical anchor: "radius ... $0.3287\pm0.0257$ ... about 20\% of the box-saturation value $\sqrt{8/3}\approx1.633$" (§5.3) | `THEORY_NOTES.md` T2′ empirical anchors §2 (0.0649→0.3287, "~20.1% of that saturation value") | Exact (rounded to "about 20%" in prose) |
| 15 | Appendix A headline tally: "52 preregistered items (34 CONFIRMED, 8 REFUTED, 5 PARTIAL, 5 UNSCOREABLE/factual)" across Stages 12–20 | Derived in this pass: `STAGE20_FINDINGS.md`'s own recount stops at Stages 12–19 (48 leg-split / 45 deliverables-excluded); Stages 12–19 baseline extended here by hand-tallying Stage 20's own W1(i)/(ii)/(iii), W2 (2 legs), W3 (deliverable), W4 (2 legs) from `STAGE20_FINDINGS.md`'s own Status table and arm text, applying the identical leg-split convention | **Derived, not pre-existing** — flagged for author spot-check; arithmetic shown in full in `main.tex` Appendix A's "Resolving the internal discrepancy" paragraph and reproduced in `REVIEW_CHECKLIST_JOURNAL.md` |

## Notes on sampling

Claims were chosen to cover every major result-bearing section of the
journal draft (rotation-mechanism verification, qubit-count replication,
real-data TPT closure, the isolation control, the amplitude-encoding
memorization finding, the comparative study's per-dataset tables, all four
generality arms, both hardware backends, and the appendix ledger) rather
than clustering in one section — a broader sweep than the ICVGIP draft's
audit, reflecting this draft's larger scope. Fourteen of fifteen claims
trace to an existing, previously-scored number in the project's findings
files with no arithmetic performed in this pass. Row 15 is the one
exception: the paper's headline "52 (34/8/5/5)" preregistration count
extends `STAGE20_FINDINGS.md`'s own "48 (12–19)" recount through this
paper's own Stage 20 arms, using the identical leg-split convention but not
itself previously stated anywhere in the repository — this is a genuinely
new computation, not a copy, and should be the first thing an author
double-checks before submission (see `REVIEW_CHECKLIST_JOURNAL.md`).

This audit does not re-verify every number in the paper — see
`REVIEW_CHECKLIST_JOURNAL.md` for what remains the responsibility of a full
author pass before submission.
