# CLAUDE.md -- Quantum Neural Collapse (QNC)

Research codebase investigating the **training dynamics** of Neural Collapse
in Variational Quantum Circuits. Du et al. (PRL 2023) proved QNC exists at
the optimum; we study how, when, and whether VQCs reach it during
gradient-based training, via quantum-information metrics (trace distance,
entanglement entropy) and controlled entanglement ablations.

## Source of truth

- **SPEC.md defines all math** (implement exactly, never simplify/substitute;
  STOP and ask if a formula seems wrong) **and section 0 defines prior work**
  (never describe QNC as "entirely open" or "first study" -- Du et al. (2023)
  proved the static optimum; we study training dynamics, geometry, and
  entanglement's role, all of which are open).
- **Read STATUS.md first in every session.** Historical playbooks and
  superseded docs live in `docs/archive/`; read them only when a task
  references them.

## Mandatory citations in any writeup or summary

Papyan, Han, Donoho (2020), PNAS (classical NC); Du, Yang, Tao, Hsieh (2023),
PRL 131, 140601 (quantum NC at optimum); San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485 (empirical QNC study).

## Hard invariants (violating any of these = broken build)

1. **Tests first.** Every function in `src/qnc/metrics.py` must have its
   SPEC section 6 unit test written and passing before it is called from any
   experiment code.
2. **Config-driven.** Every run: `python -m qnc.run configs/<name>.yaml`.
   No hyperparameter may be hardcoded in Python. Every config has a `seed`.
3. **Append-only logging.** One JSONL line per logged epoch per SPEC section 7,
   written to `results/<run_id>/metrics.jsonl`. Never overwrite a results dir;
   run_id = `<config_name>_s<seed>_<YYYYmmdd-HHMMSS>`.
4. **Figures are a pure function of metrics.jsonl.** `make_figures(run_id)`
   reads the JSONL and writes to `results/<run_id>/figures/`. Never plot from
   in-memory training state.
5. **Runtime assertions on physics:** density matrices Hermitian, trace 1,
   PSD; centered means traceless. These are `assert`s, not warnings. Use
   `numpy.linalg.eigh` for Hermitian eigenproblems, never `eig`.
6. **Exact simulation by default:** `default.qubit`, `shots: null`, unless the
   config explicitly sets shots (Stage 5 only).

## Commands

```bash
pip install -r requirements.txt        # pennylane, numpy, scipy, matplotlib, scikit-learn, torch, pytest, pyyaml
pytest -q                              # must be green before and after every task
python -m qnc.run configs/<name>.yaml  # run one experiment
python -m qnc.figures <run_id>         # regenerate figures from JSONL
python -m qnc.sweep configs/sweeps/<name>.yaml   # Stage 4+
```

## Workflow rules

- Before writing code for any task: present a short plan, wait for approval.
- After every meaningful change: run `pytest -q`. Never leave tests red.
- Commit at every green-test boundary with message `stage<N>: <what>`.
- When a stage's acceptance criteria (`PLAYBOOK_REMAINING.md`) all pass: print
  a `=== STAGE <N> COMPLETE ===` block summarizing what was built, and tell
  the user to start a fresh session with the next stage's prompt.
- If a training run behaves unexpectedly (NaN loss, no convergence, metric
  discontinuity), do NOT tweak silently -- report the anomaly with the
  relevant JSONL lines and propose diagnoses.

## Code style

Python 3.11+, type hints everywhere, numpy-style docstrings citing the SPEC
section each function implements. Pure functions for all math in
`metrics.py` -- no I/O, no globals. Small modules, each under ~300 lines.

## Do not

Install packages beyond requirements.txt without asking; use `qml.qnn`/Keras
wrappers (need explicit per-epoch hooks); delete/rewrite `results/`
directories; claim collapse/no-collapse in summaries (report numbers
instead); describe this research as "first" or "novel" without the qualifier
"first to study dynamics / trace-distance / entanglement ablation."
