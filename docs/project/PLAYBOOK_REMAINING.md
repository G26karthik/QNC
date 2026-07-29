# PLAYBOOK_REMAINING.md -- Remaining stages (16-17)

Extracted verbatim from `docs/archive/PLAYBOOK3.md` (Stages 12-15 are
complete; see STATUS.md). Same discipline: one stage per fresh session,
prompts pasted verbatim, acceptance criteria verified by you, commits at
green-test boundaries.

---

## STAGE 16 (OPTIONAL) -- Hardware validation, inference only

Prerequisite, done by YOU before the session: confirm IBM Quantum access and
current free-tier QPU minutes; install qiskit + qiskit-ibm-runtime; have an
API token in the environment. Skip this stage entirely without hardware
access; the paper stands without it.

**Paste:**

```
Read CLAUDE.md, SPEC3_ADDENDUM.md section 22, PLAYBOOK3.md Stage 16. Present a
plan first, INCLUDING an estimated total shot/job budget for approval before
any hardware submission.

Task: inference-only hardware arm per SPEC3 section 22. Load the trained n=4 L=8
R0 parameters (seed 0), transpile for the available backend, estimate z_i for
the 180 blobs3 training samples (>= 1024 shots each; batch into as few jobs
as the runtime allows), compute NC1_z, d_box, margins with binomial shot-noise
error bars, and produce a hardware-vs-simulator comparison figure and table.
Log backend name and calibration metadata. If queue or quota blocks
completion, save partial results and report exactly what ran.

Acceptance criteria: comparison table + figure for whatever fraction of
samples completed; scripts committed; pytest green (hardware code covered by
a simulator-backend mock test); commit "stage16: hardware arm". Print
=== STAGE 16 COMPLETE ===.
```

---

## STAGE 17 -- v3 assembly

**Paste:**

```
Read CLAUDE.md, SPEC section 0, SPEC2 section 10, SPEC3 section 17, PLAYBOOK3.md
Stage 17. Present a plan first.

Task: produce RESULTS_V3.md: the preregistered predictions and their scored
outcomes (Stage 12), the expressivity mechanism result (Stage 13), robustness
(Stage 14), n=6 scaling (Stage 15), hardware arm if present (Stage 16),
integrated with the Phase 1/2 narrative: the full-space nulls now framed as
one half of the measurement-subspace dichotomy. Add a THEORY_NOTES.md
skeleton listing, without proving: (T1) CE factors through the moment map z;
fibers receive no loss gradient (formal statement of the invariance);
(T2) box-constrained CE optimum lemma: with logits beta*z, z in [-1,1]^C,
the infimum over class means is at box vertices (state precisely, note beta
-> infinity limit); (T3) dimension count of fibers vs measured subspace,
n-scaling. Mark each as "to be proven by the user." Update relation-to-prior-
work: Du et al. reconciliation via measurement learnability (Stage 13), Yang
et al. fixed-classifier contrast, Mixon et al. UFM contrast. Full run_id
traceability. No claims beyond numbers. Commit "stage17: v3 assembly".
Print === STAGE 17 COMPLETE ===.
```

The theorem statements in THEORY_NOTES.md are yours to prove and write up
(T1 and T3 are short; T2 is a clean lemma). That theory section, plus the
n=6 scaling and the preregistered confirmation, is the ICML-tier package.

---

## Recovery additions

| Symptom | Action |
|---|---|
| Agent computes metrics before committing PREDICTIONS.md | Revert; preregistration order is the point. |
| Pauli projector not orthonormal for R1/R2 spans | Weight-2 Paulis are HS-orthogonal to weight-1; re-derive normalization 2^{n/2}, fix test. |
| Expressivity arms have unequal M | Re-pad depths; comparison is invalid without matching. |
| Hardware jobs stuck in queue | Save partials, proceed; the stage is optional. |
| Any stage contradicts a preregistered prediction | Report as REFUTED factually; never soften wording. |
