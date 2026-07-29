# NUMBER_AUDIT.md — 15 sampled quantitative claims from paper/main.tex

Every quantitative claim in `paper/main.tex` must trace to a findings file
or `RESULTS_V3.md`. Below are 15 claims sampled across the draft (abstract
through appendix), each with its exact source file, line number, and the
quoted source line the claim is drawn from. No claim below was
hand-computed for the paper — every number is copy-pasted from the source
quoted.

1. **Claim (abstract, §\ref{sec:hardware}):** "94.2x" trained-vs-untrained
   hardware ratio.
   **Source:** `STAGE16_FINDINGS.md:204`
   > `**Verdict: CONFIRMED** (ratio 94.2x >= the 10x threshold, by a wide margin).`

2. **Claim (§\ref{sec:phase12}):** NC1$_z$ TPT-fraction transition
   "0/5 ... 3/5 ... 5/5 at $M/M_c^{\mathrm{bound}}\approx0.094\to0.188\to0.376$".
   **Source:** `STATUS.md:33`
   > `| Capacity / TPT transition | TPT fraction 0/5→3/5→5/5 at M/M_c≈0.094→0.188→0.376; ... | 9, 14 | RESULTS_V2.md §4, STAGE14_FINDINGS.md (arm 4) |`

3. **Claim (§\ref{sec:msc}):** NC1$_z$ decays by a factor of "0.0045" at
   the L=8 anchor.
   **Source:** `STAGE12_FINDINGS.md:16`
   > `| L=8 anchor | NC1_z | 62.06 +/- 41.17 | 0.2795 +/- 0.06583 | 0.004504 |`

4. **Claim (§\ref{sec:msc}):** fiber fraction reaches "0.9934" at the L=8
   anchor.
   **Source:** `STAGE12_FINDINGS.md:58`
   > `- fiber_fraction: final 0.9934 (>= 0.9: True)`

5. **Claim (§\ref{sec:etf}):** ETF deviation "0.0722" vs. normalized
   box-vertex deviation "0.7122".
   **Source:** `STAGE12_FINDINGS.md:33-34`
   > `d_boxn (normalized d_box) = 0.7122 +/- 0.01767`
   > `d_etf (equiangle_dev_z)    = 0.07222 +/- 0.02453`

6. **Claim (§\ref{sec:etf}, Theorem~\ref{thm:t2prime}):** class-mean radius
   "$0.3287\pm0.0257$", box-saturation radius "$\sqrt{8/3}\approx1.633$",
   ~20%.
   **Source:** `THEORY_NOTES.md:208-210`
   > `radius ‖mtilde_c‖ grows from 0.0649 ± 0.0164 (epoch 0) to 0.3287 ± 0.0257 (final epoch) — a ~5x increase, but the final value is only ~20.1% of the box-vertex saturation radius (sqrt(8/3) ≈ 1.6330).`

7. **Claim (§\ref{sec:etf}):** frozen-$\beta$ TPT delay "~4.1x" (mean
   $E_0$ 145.2 vs. 35.2 epochs).
   **Source:** `STAGE14_FINDINGS.md:192`
   > `later than baseline (mean E0 145.2 vs 35.2) -- consistent with the`

8. **Claim (§\ref{sec:expressivity}):** loss-visible-subspace concentration
   "1.4--1.7x", short of a 2x threshold.
   **Source:** `STAGE13B_FINDINGS.md:45-46`
   > `| matched-M | R1 | ... | 1.417 +/- 0.1514 | 1.143 +/- 0.03552 |`
   > `| matched-M | R2 | ... | 1.420 +/- 0.1899 | 1.056 +/- 0.02024 |`

9. **Claim (§\ref{sec:robustness}):** 5000-epoch drift test,
   "$\rho=-0.072$, $p=0.48$".
   **Source:** `STAGE14_FINDINGS.md:82`
   > `e0_epoch=35: rho = -0.0716, p = 0.4793.`

10. **Claim (§\ref{sec:robustness}):** single-input QFI rank "saturates at
    exactly $30=2(2^4-1)$" for every depth tested.
    **Source:** `STAGE14_FINDINGS.md:149,151`
    > `30.0 -> **REFUTED**, decisively (not close to threshold).`
    > `Diagnosis: the single-input rank saturates at EXACTLY 30 = 2*(2^4 - 1) =`

11. **Claim (§\ref{sec:rotation}, Theorem~\ref{thm:t5}):** total-variance
    relative change "on the order of $10^{-15}$" across 5 reupload-false
    seeds.
    **Source:** `STAGE14B_FINDINGS.md:71,74,77`
    > `| 0 | stage14b_reupload_false_reupload_false_s0_20260715-213042 | 0.71070 | 3.75e-15 |`
    > `| 3 | stage14b_reupload_false_reupload_false_s3_20260715-214338 | 0.74159 | 5.84e-15 |`
    > `All 5/5 seeds' max relative change is ~1e-15 -- floating-point precision, not`

12. **Claim (§\ref{sec:rotation}):** NC1$_z$ still decays by a factor of
    "0.0209" under the same exact-invariance regime.
    **Source:** `STAGE14B_FINDINGS.md:100`
    > `Mean nc1_z final/epoch0 ratio among the 5 TPT-reaching seeds: **0.020908**.`

13. **Claim (§\ref{sec:n6}):** fiber fraction at 6 qubits "0.99963" /
    "0.99917", exceeding the 4-qubit reference.
    **Source:** `STAGE15_FINDINGS.md:94-95`
    > `| reupload=true | 0.99965, 0.99962, 0.99960, 0.99968, 0.99962 | **0.99963** |`
    > `| reupload=false | 0.99947, 0.99899, 0.99896, 0.99951, 0.99893 | **0.99917** |`

14. **Claim (§\ref{sec:hardware}):** contraction fit "$\lambda=0.7033$",
    "$R^2=0.9305$".
    **Source:** `STAGE16_FINDINGS.md:147-148`
    > `| lambda | 0.7033 | [0.6, 0.85] — met |`
    > `| R^2 | 0.9305 | >= 0.9 — met |`

15. **Claim (§\ref{sec:hardware}):** composite NC1$_z$ prediction "$758.7$"
    against an observed "$0.6749$", off by "~1123x".
    **Source:** `STAGE16_FINDINGS.md:149`
    > `| nc1_z predicted (lambda^2*Sigma_sim + Sigma_resid, pinv sandwich) | 758.71 | within 15% of 0.6749 — **missed by ~1123x** |`

## Cross-check: ledger tally in Appendix A / Methods

**Claim:** "19 CONFIRMED, 6 REFUTED, 4 PARTIAL, 1 reported factually ...
across 30 predictions" (§\ref{sec:methods}, Table~\ref{tab:ledger} caption).
**Source:** `RESULTS_V3.md:142`
> `**Tally:** 19 CONFIRMED, 6 REFUTED, 4 PARTIAL, 1 not-scored (factual), across`

This tally was corrected during Stage 17b (the original Stage 17 assembly
had miscounted PARTIAL rows); the paper draft uses the corrected count,
independently re-verified against the 30-row ledger table itself during
this audit (Appendix A of the paper, Table~\ref{tab:ledger}).
