# PLAYBOOK2.md -- Phase 2 Execution Guide
# The Overparameterization-Transition Campaign

Phase 1 is complete and committed. Phase 2 upgrades the study to a
publication-grade result targeting top-venue standards. Same session
discipline as PLAYBOOK.md: one stage = one fresh session, paste prompts
verbatim, verify acceptance criteria before advancing, commit at green-test
boundaries.

## What changed and why (read this before Stage 7)

Two literature findings reframe Phase 1's results:

1. Larocca et al. (Nat. Comput. Sci. 2023): QNN loss landscapes undergo a
   computational phase transition at a critical parameter count M_c
   (upper-bounded by the ansatz DLA dimension). Phase 1's C=3 failures ran at
   ~2% of that bound -- the failures are theory-predicted, not bad tuning.
2. Concentration of measure: unnormalized trace distance saturates near its
   ceiling for random high-dimensional states, so Phase 1's null result may be
   a metric artifact. Normalized witnesses (SPEC2 section 11) may reveal a signal
   already present in the Phase 1 logs.

Phase 2's central question: is quantum neural collapse an order parameter of
the overparameterization transition? Stage 7 reanalyzes existing data (free),
Stage 8 builds infrastructure, Stage 9 runs the headline sweep, Stage 10
generalizes, Stage 11 assembles the v2 paper.

---

## STAGE 7 -- Reanalysis of Phase 1 data (no new training runs)

**Paste:**

```
Read CLAUDE.md, SPEC.md, SPEC2_ADDENDUM.md sections 10-11, PLAYBOOK2.md Stage 7.
Present a plan first. This stage runs NO new training.

Task: recompute Phase 1's headline with normalized witnesses.
- metrics.py: add qnc1_fisher, qnc1_ratio, overlap_offdiag, purity functions
  per SPEC2 section 11, WITH the section 11.4 unit tests first
- new script src/qnc/reanalyze.py:
  (a) from JSONL alone: recompute qnc1_fisher per logged epoch for every
      Phase 1 run (uses logged qnc1_hs and class_mean_norms)
  (b) from checkpoints: for the 5-seed C=2 anchor campaign AND the 10-seed
      random-init null, reload parameters at each checkpointed epoch,
      regenerate states, compute qnc1_ratio, overlap_offdiag, purity_mean
  (c) rebuild the trained-vs-null comparison (Mann-Whitney U) for ALL FOUR
      witnesses: qnc1_trace (old), qnc1_fisher, qnc1_ratio, purity_mean
- extend figures.py: normalized-witness trajectory figure + a
  four-witness trained-vs-null comparison figure
- write REANALYSIS.md: table of trained vs null (mean+/-std, p-value) per
  witness, and a factual statement of whether any normalized witness
  separates trained from untrained where raw trace distance did not

Acceptance criteria: section 11.4 tests green; reanalysis runs on all Phase 1
run_ids without error; REANALYSIS.md and figures exist; pytest green;
commit "stage7: normalized-witness reanalysis". Print === STAGE 7 COMPLETE ===
and instruct me to start a fresh session with the Stage 8 prompt.
```

**You verify:** the four-witness comparison table. If qnc1_fisher or
purity_mean separates trained from null (p < 0.05), the Phase 1 headline
changes and the paper's story strengthens materially -- note it and carry on.
If nothing separates, that is also decision-relevant information for Stage 9's
predictions. Either way, do not reinterpret Phase 1 in writing yet; Stage 11
does that with the full picture.

---

## STAGE 8 -- Phase 2 infrastructure

**Paste:**

```
Read CLAUDE.md, SPEC.md, SPEC2_ADDENDUM.md sections 12-16, PLAYBOOK2.md Stage 8.
Present a plan first.

Task: build everything Stage 9 needs, tests first.
- data.py: add blobs3, bars_stripes_4x4, linearly_separable_4d generators per
  SPEC2 section 13, deterministic per seed, with shape/balance unit tests
- models.py: add amplitude encoding option (qml.AmplitudeEmbedding,
  normalize=True) gated by config `encoding`; add config `loss: ce|mse` with
  MSE per SPEC2 section 14; n_params reported by the model object
- metrics.py: add qfi_matrix and qfi_rank per SPEC2 section 12.2 with BOTH unit
  tests (RY 1-qubit value 0.25; redundant-parameter rank 1) and the
  autograd-vs-parameter-shift agreement test on one small config
- logging_utils.py: extend schema with SPEC2 section 15 fields
- train.py: log n_params, m_over_mc every line; compute qfi_rank at epoch 0
  and final epoch only
- configs/stage9_transition.yaml: n=4, C=3, dataset blobs3 (dist 3.0, std 0.8,
  60/class), reupload true, entangling true, CE loss, beta0 5.0, lr 0.05,
  epochs 600, tpt_multiple 5, base for the L sweep
- sweep config configs/sweeps/transition_L.yaml: L in {2,4,8,12,16,21,26,32},
  seeds {0..4}
- smoke test: run L=2 and L=21 for 50 epochs each, confirm no NaN, schema
  fields populated, qfi_rank computed at epoch 0

Acceptance criteria: all new unit tests green (>= 8 new tests); smoke runs
logged with complete new schema; pytest green; commit "stage8: phase2 infra".
Print === STAGE 8 COMPLETE === and instruct me to start a fresh session with
the Stage 9 prompt.
```

**You verify:** open a smoke-run JSONL line and check n_params, m_over_mc,
qfi_rank are present and sane (L=2 => n_params=24, m_over_mc=24/255~0.094;
qfi_rank <= 24). If qfi computation is slow, confirm it only runs at epoch 0
and final.

---

## STAGE 9 -- The transition sweep (headline experiment)

**Paste:**

```
Read CLAUDE.md, SPEC2_ADDENDUM.md section 12, PLAYBOOK2.md Stage 9. Present a plan
first.

Task: run the overparameterization-transition campaign.
- execute configs/sweeps/transition_L.yaml: 8 L-values x 5 seeds = 40 runs at
  n=4, C=3, blobs3. Run sequentially; estimated wall time is hours, so run in
  background and monitor; report progress after each L-value completes
- extend report.py: per-L table -- TPT fraction, E0 mean, final qnc1_fisher,
  qnc1_ratio, overlap_offdiag, purity_mean, equiangle_dev, entropy_mean,
  qfi_rank(final) -- mean+/-std across seeds
- extend figures.py: the order-parameter figure per SPEC2 section 12.3 -- all
  outcome variables vs M/M_c_bound, shared x-axis, qfi_rank saturation point
  marked with a vertical line
- statistics: Mann-Whitney U comparing final qnc1_fisher and equiangle_dev
  between the underparameterized group (L<=12) and overparameterized group
  (L>=21), pooling seeds

Acceptance criteria: 40/40 runs logged; order-parameter figure exists;
report table complete; pytest green; commit "stage9: transition sweep".
Describe results factually only (which quantities changed at which L, p-values)
-- no "confirms/refutes" language. Print === STAGE 9 COMPLETE === and instruct
me to start a fresh session with the Stage 10 prompt.
```

**You verify (this is the money plot):** three specific things on the
order-parameter figure. (1) Does TPT fraction jump from ~0 to ~1 somewhere in
the sweep? (2) Does that jump align with qfi_rank saturation? (3) Do
equiangle_dev and qnc1_fisher improve sharply at the same point, or gradually,
or not at all? Sharp+aligned = strongest possible result (collapse gated by
the transition). Gradual = still publishable (collapse tracks capacity, not
the transition). No TPT even at L=32 = stop, message me before Stage 10; we
diagnose (first suspects: blob_dist too small, lr schedule, beta ceiling).

---

## STAGE 10 -- Generalization: datasets, loss, entanglement x transition

**Paste:**

```
Read CLAUDE.md, SPEC2_ADDENDUM.md sections 13-14, PLAYBOOK2.md Stage 10. Present a
plan first.

Task: three targeted extensions of the Stage 9 result, 3 seeds each unless
stated. Pick L_under = largest L with TPT fraction ~0 and L_over = smallest L
with TPT fraction ~1 from Stage 9's table.
1. Dataset generality at L_over: bars_stripes_4x4 (both angle and amplitude
   encoding), linearly_separable_4d, mnist_pca4 (C=3). Same metrics.
2. Loss arm: matched CE vs MSE at L_over on blobs3, 5 seeds each. MSE is the
   loss Du et al.'s theorem actually covers -- report all collapse witnesses
   side by side; this is the direct theory comparison.
3. Entanglement x transition interaction: entangling=false at BOTH L_under and
   L_over (blobs3, 3 seeds each). Note: removing entanglers changes the DLA --
   the non-entangling ansatz has DLA dim 3n=12 regardless of L, so it is
   ALWAYS effectively saturated and can never exceed su(2)^n capacity. Log
   m_over_mc against this corrected bound (12) for the ablation arms; this
   nuance goes in the figure caption and RESULTS.
- report.py and figures.py extended accordingly (CE-vs-MSE figure; ablation
  figure with corrected bounds)

Acceptance criteria: all three extension families logged with figures and
tables; pytest green; commit "stage10: generalization arms". Print
=== STAGE 10 COMPLETE === and instruct me to start a fresh session with the
Stage 11 prompt.
```

**You verify:** the CE-vs-MSE comparison especially. If MSE shows cleaner
collapse (classical literature hints it might, and Du's theorem is MSE), that
is a headline-adjacent finding. Also check the ablation: if entangling=false
fails TPT even at L_over params, the Phase 1 entanglement finding survives
the overparameterization reframing -- and gains a DLA-based explanation
(capacity ceiling of 12), which resolves it elegantly.

---

## STAGE 11 -- v2 paper assembly

**Paste:**

```
Read CLAUDE.md, SPEC.md section 0, SPEC2_ADDENDUM.md section 10, PLAYBOOK2.md
Stage 11. Present a plan first.

Task: produce RESULTS_V2.md assembling the full Phase 1 + Phase 2 evidence:
- methodology (cite SPEC/SPEC2 sections; mandatory citations from SPEC section 0.4
  plus SPEC2 section 10)
- the Stage 7 reanalysis (how normalized witnesses changed or confirmed the
  Phase 1 null)
- the Stage 9 order-parameter result with the headline figure
- Stage 10 generalization: datasets, CE-vs-MSE, entanglement-x-transition with
  the DLA capacity explanation
- an explicit "relation to prior work" section: Du et al. (what their theorem
  predicts and what our MSE arm shows), Larocca et al. (predicted transition
  vs our measured qfi_rank saturation and TPT jump), Bowles et al. (their
  entanglement-dispensability finding vs our interpolation-necessity finding),
  San Sebastian et al.
- limitations (simulator-only, n=4 primary, single bipartition, DLA bound is
  an upper bound not exact M_c, blobs3 is synthetic)
- full run_id traceability per figure
No claims beyond the numbers. Commit "stage11: v2 results assembly".
Print === STAGE 11 COMPLETE ===.
```

You write the final abstract, introduction, and discussion yourself, as
before. The agent assembles evidence; the claims are yours.

---

## Venue plan (for you, not the agent)

Timeline reality: ICML 2027 submission will be ~Jan 2027. Working backward:
- Now -> +4 weeks: Stages 7-10 (Stage 9 is hours of compute, not days)
- +6 weeks: v2 draft, post to arXiv to timestamp the claim
- Fall 2026: submit to QTML 2026 and a NeurIPS 2026 workshop (feedback +
  visibility + a line on the record)
- In parallel: cold-email 2-3 researchers whose gaps this fills (Du/Hsieh
  orbit; San Sebastian/Orus group; NC-theory people like Papyan's or
  Thrampoulidis's students) with the arXiv link and one specific figure --
  the order-parameter plot. A senior collaborator raises the ICML ceiling
  more than any experiment.
- Jan 2027: ICML main track ONLY if Stage 9 produced a sharp, aligned
  transition AND you have added either (a) a theory section (even a
  UFM-style toy analysis of why collapse requires M >= M_c) or (b) a second
  qubit count (n=6 partial sweep) showing the effect scales. Otherwise
  PRX Quantum / Quantum journal, which is where this literature actually
  lives and where Du et al.-adjacent empirical work gets read.

## Recovery additions (extends PLAYBOOK.md table)

| Symptom | Action |
|---|---|
| Stage 9: no TPT even at L=32 | Stop. Raise blob_dist to 4.0 (one run to confirm), then message me -- do not silently rescope. |
| QFI computation dominates wall time | Confirm epoch-0/final-only; reduce QFI input batch to 8; never skip it. |
| Agent conflates M_c bound with exact M_c | Point to SPEC2 section 12.1: 4^n-1 is an UPPER BOUND; the empirical locator is qfi_rank saturation. |
| Ablation arms logged with wrong m_over_mc | Non-entangling DLA bound is 12, per Stage 10 prompt. |
