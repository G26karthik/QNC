# PLAYBOOK3.md -- Phase 3 Execution Guide
# The Measurement-Subspace Collapse Campaign

Phase 2 is committed. Phase 3 tests one reconciling hypothesis with a
preregistered prediction set, then hardens it. Same discipline: one stage per
fresh session, prompts pasted verbatim, acceptance criteria verified by you,
commits at green-test boundaries.

The hypothesis in one line: collapse happens where the loss can see (the
C-dimensional measurement subspace) and nowhere else; full-space witnesses
were dominated by loss-invisible fiber directions. If Stage 12 confirms it,
the paper's headline becomes a new phenomenon (measurement-subspace collapse
with box-vertex geometry) plus a mechanism, not a null.

Stage order and cost: 12 (free, existing checkpoints), 13 (the expressivity
sweep, ~20 runs), 14 (robustness, ~25 runs), 15 (n=6 scale, ~17 runs),
16 (optional hardware, inference only), 17 (v3 assembly). 12 is decisive:
its outcome determines how much of 13-15 matters, so ALWAYS review Stage 12
results with me before launching Stage 13.

---

## STAGE 12 -- Preregistered reanalysis: does z-space collapse? (no new training)

**Paste:**

```
Read CLAUDE.md, SPEC.md, SPEC2_ADDENDUM.md, SPEC3_ADDENDUM.md sections 17-18,
PLAYBOOK3.md Stage 12. Present a plan first. This stage trains nothing.

Task A (before ANY computation): create PREDICTIONS.md preregistering, for
the Phase 2 L=8 anchor (5 seeds) and the Phase 1 C=2 anchor:
- NC1_z and r_z: predicted to DECAY by >= 1 order of magnitude post-TPT
- beta * mean_margin: predicted to GROW post-TPT
- d_box vs equiangle_dev_z at C=3: means predicted to approach BOX VERTICES
  (d_box decreasing) more closely than ETF geometry
- fiber decomposition: V_meas predicted to decay; V_fiber predicted to stay
  at null level; fiber_fraction predicted -> ~1
- untrained null (10 seeds): NC1_z predicted NOT to decay (sanity arm)
Commit PREDICTIONS.md separately BEFORE implementing metrics.

Task B: implement SPEC3 section 18 in metrics.py, tests first (section 18.5, >= 5 new
tests), including the fiber decomposition with the Pauli orthonormalization
check.

Task C: extend reanalyze.py to recompute, from existing checkpoints:
all section 18 metrics per checkpointed epoch for (i) Phase 2 L=8 anchor seeds,
(ii) the full Stage 9 L sweep final epochs, (iii) Phase 1 C=2 anchor (d_box,
NC1_z, margins only), (iv) the 10-seed untrained null. Extend figures.py:
z-space collapse trajectories, box-vs-ETF comparison, and the headline
two-panel figure V_meas vs V_fiber over training.

Task D: write STAGE12_FINDINGS.md scoring every preregistered prediction as
CONFIRMED / REFUTED / PARTIAL with the numbers, including Mann-Whitney U of
trained-vs-null NC1_z.

Acceptance criteria: PREDICTIONS.md committed before metrics code; all new
tests green; findings document complete with per-prediction scoring; figures
exist; pytest green; commit "stage12: measurement-subspace reanalysis".
Print === STAGE 12 COMPLETE === and instruct the user to review
STAGE12_FINDINGS.md with their advisor/assistant BEFORE Stage 13.
```

**You verify, and then stop and talk to me.** The four possible worlds:
1. z-collapse confirmed + fiber flat: hypothesis confirmed, full speed ahead,
   the paper is now "Measurement-Subspace Neural Collapse in VQCs."
2. z-collapse confirmed + box beats ETF: add the geometry claim to the title
   ("collapse follows measurement geometry, not ETF").
3. NC1_z does NOT decay: the models interpolate via margins without even
   z-space concentration. That is its own striking finding (margin growth
   without any collapse) and redirects Stage 13; do not launch it unmodified.
4. Mixed across seeds/depths: we scope claims by regime.
Bring me the findings file either way.

---

## STAGE 13 -- Measurement-expressivity sweep (the mechanism experiment)

**Paste:**

```
Read CLAUDE.md, SPEC3_ADDENDUM.md section 19, PLAYBOOK3.md Stage 13. Present a
plan first.

Task: implement readout families R1, R2, R3 per SPEC3 section 19 with parameter-
count matching across arms (document the matched M and how padding was done
in the config comments; log m_params_measurement). Generalize the section 18.4
projector to each arm's measured Pauli span, with an orthonormalization unit
test per arm. Run R0-R3, 5 seeds each, blobs3 C=3 n=4, 600 epochs. Extend
report.py and figures.py: expressivity-axis figure -- full-space witnesses
(qnc1_fisher, purity_mean, overlap_offdiag) AND fiber_fraction vs readout
family, plus z/feature-space NC1 per arm.

Acceptance criteria: 20 runs logged; orthonormalization tests green per arm;
expressivity figure exists; factual per-arm table with Mann-Whitney U
R0-vs-R3 on full-space witnesses; pytest green; commit "stage13:
expressivity sweep". Print === STAGE 13 COMPLETE === and instruct the user
to start a fresh session with the Stage 14 prompt.
```

**You verify:** the expressivity figure's slope. Full-space witnesses
strengthening monotonically R0 to R3 completes the reconciliation story with
Du et al. (measurement learnability is the interpolating knob). Flat would
mean the fiber story needs revision; report either factually.

---

## STAGE 14 -- Robustness of the dichotomy

**Paste:**

```
Read CLAUDE.md, SPEC3_ADDENDUM.md section 20, PLAYBOOK3.md Stage 14. Present a
plan first.

Task: run the five robustness arms of SPEC3 section 20 exactly:
(1) beta_max=500, 5 seeds; (2) one 5000-epoch long-horizon run with Spearman
drift test on full-space witnesses; (3) label smoothing 0.1, 5 seeds;
(4) transition localization L in {3,5,6}, 10 seeds each; (5) single-input
QFI rank fix at L in {8,16,21,26,32}. Extend report.py with a robustness
table: for arms 1-3, full-space witnesses AND section 18 z-space witnesses vs the
Stage 13 R0 baseline, with tests. For arm 4, the refined TPT-fraction curve.
For arm 5, single-input ranks vs the batch-aggregated ranks of Phase 2.

Acceptance criteria: all arms logged; robustness table and refined transition
figure exist; pytest green; commit "stage14: robustness". Print
=== STAGE 14 COMPLETE === and instruct the user to start a fresh session
with the Stage 15 prompt.
```

**You verify:** arms 1-3 are the referee-proofing. If beta=500 or label
smoothing moves full-space witnesses materially, the claim gets rescoped
honestly (and that is itself informative: collapse pressure CAN reach the
fibers under X). The long-horizon Spearman test converts "no collapse" into
"no collapse and no drift over 12x longer training," which is the sentence a
reviewer needs.

---

## STAGE 15 -- Scale replication at n=6

**Paste:**

```
Read CLAUDE.md, SPEC3_ADDENDUM.md section 21, PLAYBOOK3.md Stage 15. Present a
plan first.

Task: blobs3 in R^6, n=6, C=3. Depth probe L in {8,16,32,64} (3 seeds) to
find the smallest interpolating depth; then 5 seeds there with the full
witness set (SPEC2 + SPEC3 section 18). Report wall-time per run after the first
completes; if a full run exceeds ~2h, consult the user before continuing
rather than silently reducing scope. Figures: the n=6 V_meas/V_fiber panel
beside the n=4 one; fiber_fraction n=4 vs n=6.

Acceptance criteria: probe + campaign logged; side-by-side dichotomy figure
exists; pytest green; commit "stage15: n6 replication". Print
=== STAGE 15 COMPLETE === and instruct the user to start a fresh session
with the Stage 16 prompt (optional hardware) or Stage 17 (assembly).
```

**You verify:** fiber_fraction at n=6 vs n=4. The dimension-counting argument
predicts it moves closer to 1 as n grows (fibers grow much faster than the
measured subspace). A clean scaling direction turns one observation into a
trend, which is what "will it scale?" reviewers ask.

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
