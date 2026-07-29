# PLAYBOOK.md -- Stage-by-Stage Execution Guide

How to drive this project with Claude Code so every stage is verified before
the next begins. **One stage = one fresh session.** Never continue a completed
stage's session into the next stage -- a fresh session with a tight prompt keeps
the agent anchored to SPEC.md instead of its own earlier improvisations.

## Important: research positioning

This project builds on Du et al. (PRL 2023), who proved that the global
optimum of quantum classifiers exhibits ETF structure (quantum Neural
Collapse). Our contribution is studying the **training dynamics** -- how
and when VQCs reach (or fail to reach) that optimum -- using
quantum-native metrics (trace distance, entanglement entropy) and
controlled entanglement ablations. See SPEC.md section 0 for full details.

Never frame prompts or outputs as if QNC is "entirely open" or "first
studied here." The static result exists; the dynamics, density-matrix
metrics, and entanglement story are what's new.

## Session hygiene (applies to every stage)

1. `cd` into this folder, run `claude`.
2. Paste the stage prompt verbatim (they all force plan mode via "present a
   plan first").
3. Review the plan. Approve only if it references the correct SPEC sections.
4. At stage end the agent prints `=== STAGE N COMPLETE ===` and tells you to
   open a new session -- verify the acceptance checklist yourself first, then
   `/exit`, then start the next session.
5. If anything looks wrong mid-stage: `Esc` to interrupt, correct course, or
   `/clear` and re-paste the stage prompt (git protects finished work).

---

## STAGE 0 -- Scaffolding

**Paste:**

```
Read CLAUDE.md, SPEC.md and PLAYBOOK.md fully. We are in Stage 0.

Task: scaffold the repository exactly as follows, then present a plan before
writing anything:
- git init, .gitignore (results/, __pycache__, *.ckpt, .venv)
- requirements.txt: pennylane, numpy, scipy, matplotlib, scikit-learn, torch,
  pytest, pyyaml
- package skeleton src/qnc/{__init__,data,models,metrics,train,logging_utils,
  figures,run,sweep}.py with docstring stubs only (no logic yet)
- tests/test_metrics.py containing ALL unit tests from SPEC section 6, written
  against the (not yet implemented) functions in metrics.py, marked
  xfail-until-implemented is NOT allowed -- instead implement metrics.py NOW,
  test-first, function by function: trace_distance, hs_inner, hs_norm,
  class_means, centered_means, qnc1_metrics, qnc2_metrics, reduced_density,
  von_neumann_entropy, classical_nc1, classical_nc2
- logging_utils.py: JSONLLogger implementing SPEC section 7 schema exactly, plus
  run_id generation per CLAUDE.md
- verify: pytest -q fully green, then commit "stage0: scaffold + verified metrics"

Acceptance criteria: all SPEC section 6 tests pass; pytest count >= 12; no experiment
code exists yet. When done, print === STAGE 0 COMPLETE === and instruct me to
start a fresh session with the Stage 1 prompt.
```

**You verify before moving on:** run `pytest -q` yourself; open
`tests/test_metrics.py` and spot-check that the Bell-state entropy test expects
exactly 1.0 and the ETF test expects cosines of -0.5. If the agent wrote
tolerance-999 tests or skipped a test, make it fix that now.

---

## STAGE 1 -- Classical NC control experiment

Purpose: validate the *instrument* on a phenomenon known to exist. If NC1
doesn't decay here, the bug is ours, not physics.

**Paste:**

```
Read CLAUDE.md, SPEC.md sections 0, 5, 7, 8, PLAYBOOK.md Stage 1. Present a plan first.

Task: implement the classical control experiment.
- data.py: MNIST loader, select classes per config, samples_per_class per
  config, deterministic split by seed
- models.py: MLP per SPEC section 5 (784-512-512-64-C, ReLU)
- train.py: SGD momentum 0.9, weight_decay 5e-4, cross-entropy, epoch hook
  that extracts penultimate features on the full train set, computes classical
  NC1/NC2 via the ALREADY-TESTED functions in metrics.py, logs per SPEC section 7
  (quantum fields = null), checkpoints per CLAUDE.md
- figures.py: implement SPEC section 8 figures 1-3 (skip entropy/beta for classical)
- configs/stage1_classical.yaml: C=3 (digits 0,1,2), 500/class, 350 epochs,
  lr 0.05, log_interval per SPEC section 7, seed 0
- run it: python -m qnc.run configs/stage1_classical.yaml
- then rerun with seeds 1 and 2 (edit seed via CLI override, add --seed flag)

Acceptance criteria:
- train acc hits 100% and training continues >=5x past that epoch
- NC1 decays by >=2 orders of magnitude from its early-training peak
- equiangle_dev trends toward 0 and pairwise cosines approach -0.5 (C=3)
- figures exist for every run and visibly show the collapse
- pytest still green; commit "stage1: classical NC reproduced"
Report the final NC1, equinorm_cv, equiangle_dev values for all 3 seeds in a
table. Print === STAGE 1 COMPLETE === and instruct me to start a fresh session
with the Stage 2 prompt.
```

**You verify:** open `results/*/figures/qnc1.png` -- NC1 must fall roughly
monotonically after E_0 on the log axis. Cosine trajectories must converge
toward the dashed -0.5 line. If not, debug HERE before any quantum work.

**Research knobs if collapse is weak** (this is real ML work, in order of
impact): train longer (NC needs deep TPT -- try 500+ epochs); raise weight
decay (5e-4 -> 1e-3; weight decay is known to strengthen NC); lower lr with a
step decay at 1/3 and 2/3 of training; increase samples/class. Change one
knob per run, keep every run's JSONL.

---

## STAGE 2 -- VQC classifier reaching TPT

**Paste:**

```
Read CLAUDE.md, SPEC.md sections 0, 2, 3, 7, 8, PLAYBOOK.md Stage 2. Present a plan first.

Task: implement the quantum model and get it to zero train error.
- data.py: add PCA pipeline -- fit PCA(n_components=n_qubits) on train split,
  standardize, scale to [-pi, pi] per SPEC section 2, fully deterministic per seed
- models.py: add VQC per SPEC section 2: angle encoding, StronglyEntanglingLayers(L),
  optional reupload flag, optional entangling=false ablation ansatz, Pauli-Z
  readout on first C qubits, learnable beta initialized from config,
  default.qubit exact statevector, torch interface
- train.py: extend for VQC -- Adam (lr from config), per-epoch hook extracts
  |psi_i> for the full train set via qml.state, builds rho_i, computes ALL
  SPEC section 4 metrics using the tested metrics.py functions, logs full SPEC section 7
  schema including beta and entropy fields
- smoke config configs/stage2_smoke.yaml: two-moons dataset (add to data.py),
  n=4 qubits, C=2, L=3, 300 epochs, lr 0.05, beta0 5.0, seed 0 -- run it and
  confirm 100% train acc and sane metric values (no NaN, entropy in [0,2])
- main config configs/stage2_vqc_base.yaml: MNIST digits 0,1,2 (C=3),
  n=6 qubits, L=4, reupload true, 200 samples/class, 400 epochs, lr 0.05,
  beta0 5.0, seed 0 -- run it

Acceptance criteria:
- smoke run reaches 100% train acc; main run reaches 100% train acc with
  tpt_reached=true and continues >=5x E0
- all SPEC section 4 runtime assertions held (no assertion errors in logs)
- figures 1-6 from SPEC section 8 generated for both runs
- pytest green; commit "stage2: VQC reaches TPT"
Do NOT interpret the NC metrics yet. Report E0, final train/test acc, final
beta. Print === STAGE 2 COMPLETE === and instruct me to start a fresh session
with the Stage 3 prompt.
```

**You verify:** `training.png` shows train acc pinned at 1.0 for the long tail.
Check beta grew (if beta stays ~5 and loss plateaus above ~0.1, the logits are too
soft to collapse).

**Research knobs -- this stage is where the actual ML tuning lives.** If the
VQC can't interpolate (train acc < 100%), there is no TPT and no experiment,
so tune in this order:
1. **beta_0 (logit temperature)** -- most underrated knob. Try 1, 5, 10; learnable
   vs fixed. Too small -> cross-entropy floor, no collapse pressure. Too large
   early -> gradient saturation. (Note: independently validated by QMT paper,
   arXiv:2606.22551)
2. **Capacity:** L = 2 -> 4 -> 6; reupload on (usually decisive on real data).
3. **Optimizer:** Adam lr 0.1 / 0.05 / 0.01; cosine decay over last half.
4. **Problem difficulty:** drop to C=2, or easier digit triplet (0,1,7), or
   fewer samples/class (150) -- interpolation first, scale up later.
5. **Qubits n:** 6 -> 8 raises capacity but watch barren-plateau symptoms
   (gradient norm ~0 from epoch 0 -- log grad norms if suspected).
Never tune more than one knob per run. Every attempt stays in results/.

---

## STAGE 3 -- Quantum NC dynamics measurement campaign

**Paste:**

```
Read CLAUDE.md, SPEC.md sections 0, 4, 8, PLAYBOOK.md Stage 3. Present a plan first.

Task: the headline experiment -- studying training DYNAMICS of QNC. Du et al.
(PRL 2023) proved the static optimum; we are measuring the trajectory.
Using the Stage 2 base config as anchor:
- add --seed override support if not present; run stage2_vqc_base with seeds
  0,1,2,3,4 (5 runs)
- extend figures.py: multi-seed aggregation (mean +/- 1 std band) per SPEC section 8,
  and the gram_triptych at init/E0/final
- add a small analysis script src/qnc/report.py that reads all 5 JSONLs and
  prints a markdown table: E0, final qnc1_trace, qnc1_hs, equinorm_cv,
  equiangle_dev, entropy_mean per seed + mean+/-std

Acceptance criteria:
- 5/5 runs tpt_reached=true (if any fails, report it and stop -- we retune
  in this session before proceeding)
- aggregated figures exist; triptych exists
- pytest green; commit "stage3: 5-seed QNC dynamics campaign"
Report the table. Describe the metric trajectories factually (increasing/
decreasing/flat, by how much) without claiming collapse or its absence.
Print === STAGE 3 COMPLETE === and instruct me to start a fresh session with
the Stage 4 prompt.
```

**You verify (this is the science):** look at `qnc1.png` -- does within-class
trace distance decay after E_0, and do both witnesses agree? Look at cos_theta
trajectories vs the -0.5 line. Three honest outcomes: clear collapse, clear
absence, or partial (e.g., QNC1 decays but geometry never reaches ETF). All
three are reportable results.

---

## STAGE 4 -- Sweeps and the entanglement ablation

**Paste:**

```
Read CLAUDE.md, SPEC.md, PLAYBOOK.md Stage 4. Present a plan first.

Task: implement sweep.py driving multiple configs sequentially with a
sweep YAML (list of config overrides), then create and run these sweeps,
3 seeds each:
- ablation_entangling: base vs entangling=false  (the key scientific ablation --
  does collapse require entanglement?)
- sweep_qubits: n = 4, 6, 8 (C=3 fixed)
- sweep_depth: L = 2, 4, 6
- sweep_classes: C = 2, 3, 4 (etf_target changes: -1, -0.5, -1/3 -- verify
  figures use the per-config target)
- sweep_wd: weight_decay = 0, 1e-4, 1e-3
Add comparison figures: one panel per sweep overlaying final-epoch
equiangle_dev and qnc1_trace across the swept values.

Acceptance criteria: all runs logged, comparison figures exist, report.py
extended to a cross-config table, pytest green, commit "stage4: sweeps".
Print === STAGE 4 COMPLETE === and instruct me to start a fresh session with
the Stage 5 prompt.
```

**You verify:** the entangling=false column -- if collapse appears only with
entanglement, entropy dynamics become the core of your paper's story; if it
appears in both, the story is that collapse is encoding/loss-driven. Either
way you have a finding.

---

## STAGE 5 -- Robustness, controls, statistics

**Paste:**

```
Read CLAUDE.md, SPEC.md, PLAYBOOK.md Stage 5. Present a plan first.

Task: falsification checks on the best configuration from Stage 4:
1. shots run: identical config but shots=1000 (metrics still computed from
   analytic state -- shots affect TRAINING only; note this in the config docs).
   Question: does the geometry survive noisy optimization?
2. shuffled-labels control: identical config, labels randomly permuted
   (seeded). Expect: if train acc reaches 100% (memorization), NC metrics
   should behave differently from the real-label runs -- a converging model
   alone must not trivially produce our signal.
3. random-init control: compute all SPEC section 4 metrics at epoch 0 across 10
   random seeds without training -- the null distribution of equiangle_dev and
   qnc1 under no learning.
4. statistics in report.py: for real vs shuffled and entangled vs not, report
   mean+/-std of final metrics and a Mann-Whitney U p-value (scipy) across seeds.

Acceptance criteria: 3 experiment families logged with figures, statistical
table produced, pytest green, commit "stage5: controls".
Print === STAGE 5 COMPLETE === and instruct me to start a fresh session with
the Stage 6 prompt.
```

---

## STAGE 6 -- Writeup assets

**Paste:**

```
Read CLAUDE.md, PLAYBOOK.md Stage 6. Present a plan first.

Task: produce RESULTS.md assembling: methodology summary (cite SPEC sections
and mandatory references from SPEC section 0), the cross-config table, all headline
figures with 2-3 sentence factual captions, the controls comparison, and a
limitations section (simulator-only, small n, single bipartition for entropy,
single encoding family, comparison to Du et al.'s theoretical predictions).
List every run_id used per figure for full traceability. No claims beyond the
numbers. Frame contributions per SPEC section 0.3: dynamics, trace distance,
entanglement ablation. Commit "stage6: results assembly".
Print === STAGE 6 COMPLETE ===.
```

You then write the actual interpretation/abstract yourself -- the agent
assembles evidence; the scientific claims are yours.

---

## Recovery playbook

| Symptom | Action |
|---|---|
| Agent invents a formula variant | Point it to the SPEC section: "Re-read SPEC section X and re-implement exactly." |
| Agent claims QNC is "novel" or "first" | Point it to SPEC section 0 and CLAUDE.md citation rules. |
| Tests weakened (huge tolerances) | "Restore SPEC section 6 tolerances; fix the code, not the test." |
| Loss NaN | Check beta explosion first (clamp beta <= 50), then lr. |
| Train acc stuck < 100% | Stage 2 knob list, one change per run. |
| Metric discontinuity between epochs | Almost always a checkpoint/state-extraction bug -- diff rho_bar_c across the jump. |
| Session drifting / confused | /clear, re-paste the stage prompt; git has your back. |
