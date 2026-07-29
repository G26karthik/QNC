# Quantum Neural Collapse: Collapse by Rotation

**Does Neural Collapse — the terminal-phase geometric regularity of trained
deep classifiers — occur when a Variational Quantum Circuit (VQC) is
actually trained, and if so, how?**

Papyan, Han, and Donoho (2020) established Neural Collapse (NC) for
classical deep networks. Du, Yang, Tao, and Hsieh (2023) proved a quantum
analogue holds at the *static global optimizer* of a regularized quantum
classifier — but said nothing about whether gradient-based training ever
reaches it, or by what route. This project answers that question
empirically and theoretically: it tracks collapse metrics epoch-by-epoch
across training, proves a structural theorem about the mechanism, and
verifies it to machine precision on synthetic data, real vision data
(MNIST, Fashion-MNIST), three qubit counts, and two independent IBM
quantum hardware backends.

## Headline finding: collapse by rotation

Cross-entropy loss only ever reads a `C`-dimensional measurement
projection of the quantum state. Every **full-Hilbert-space** collapse
witness we compute stays statistically indistinguishable from an untrained
baseline — a null result. But restricting to the measurement-visible
projection recovers a large, real collapse signal (2–3 orders of magnitude
decay). We prove and verify (Proposition 3 in the paper) that for a broad,
practically relevant architecture class this is **not a contraction of the
state constellation — it is an exact rotation**: training conjugates every
sample's state by a single global unitary, which is a Hilbert–Schmidt
isometry. We verify this to machine precision (~10⁻¹⁵ relative change) on:

- a controlled synthetic benchmark, at three qubit counts (`n = 4, 6, 8`)
- real vision data (MNIST, Fashion-MNIST) under two independent encodings
  (angle embedding, amplitude embedding)
- a second optimizer (SGD) and a second ansatz family (hardware-efficient)
- two independent IBM hardware backends (`ibm_marrakesh`, `ibm_kingston`),
  inference-only

A matched classical MLP baseline shows the *opposite* absolute-scale
behavior: its within-class scatter **grows** 27–163× under training even
as its normalized collapse ratio falls — a distinction invisible to
normalized diagnostics alone, and evidence the quantum invariance is
structural, not an artifact of weak regularization.

Every prediction in this campaign was preregistered (threshold committed
before the metric was computed) and scored CONFIRMED/REFUTED/PARTIAL —
refuted and partial results are reported plainly, not hidden. Full ledger:
52 items across ten experimental stages, itemized in the paper's
Appendix A.

## The paper

**[`paper/main.pdf`](paper/main.pdf)** — "Collapse by Rotation:
Training-Dynamics Neural Collapse in Variational Quantum Circuits,
Verified on Real Vision Data," prepared for submission to Springer's
*Quantum Machine Intelligence*. Source: `paper/main.tex` (compiles with
`pdflatex` → `bibtex` → `pdflatex` ×2 using the included Springer Nature
`sn-jnl` class). Self-audit documents: `paper/NUMBER_AUDIT_JOURNAL.md`
(15 sampled claims traced to source) and
`paper/REVIEW_CHECKLIST_JOURNAL.md`.

## Repository map

```
QNC/
├── paper/              The paper: main.tex, main.pdf, figures, audit docs
├── src/qnc/             Core library: models, metrics, training, figures, hardware
├── tests/                173+ unit/integration tests, one per metrics.py function
├── configs/              YAML experiment configs (one seed, one config, one run)
├── scripts/              Stage-specific analysis/scoring scripts
├── results/               Tracked hardware run outputs (backs the hardware claims)
├── figs/                  Standalone figures (mechanism schematic, scaling plots)
└── docs/
    ├── spec/              SPEC.md + addenda — the frozen mathematical specification
    ├── theory/            THEORY_NOTES.md — theorem statements and proofs
    ├── preregistration/   PREDICTIONS.md — every threshold, committed before results
    ├── findings/          Per-stage findings records + results assemblies
    ├── references/        Citation list
    ├── project/           CLAUDE.md, STATUS.md, and other project bookkeeping
    └── archive/           Superseded planning documents
```

## Reproducing the results

```bash
pip install -r requirements.txt
pytest -q                                    # must be green: 173+ tests
python -m qnc.run configs/<name>.yaml        # run one experiment (config-driven, seeded)
python -m qnc.figures <run_id>               # regenerate figures from the run's metrics.jsonl
python -m qnc.sweep configs/sweeps/<name>.yaml
```

Exact quantum simulation (`default.qubit`, no shot noise) is the default
throughout; hardware runs are inference-only against saved trained
checkpoints, gated on a translation-fidelity check (<10⁻⁶ vs. the exact
simulator) before any hardware submission.

## Scientific rigor: how correctness is enforced

- **Frozen specification.** [`docs/spec/SPEC.md`](docs/spec/SPEC.md) and its
  addenda define every formula, unit-test value, and logging schema.
  Code implements it exactly; nothing is simplified or substituted without
  the spec changing first.
- **Preregistration.** [`docs/preregistration/PREDICTIONS.md`](docs/preregistration/PREDICTIONS.md)
  records every threshold *before* the corresponding metric is computed.
  Findings are scored strictly against that record — see
  [`docs/findings/`](docs/findings/) for every stage's results.
- **Tests before code.** Every function in `src/qnc/metrics.py` has its
  spec-defined analytic unit test written and passing before it is ever
  called from experiment code.
- **Runtime physics assertions.** Density matrices are asserted Hermitian,
  trace-1, and positive semidefinite at runtime, not just checked in tests.
- **Append-only, config-driven logging.** Every run is a seeded YAML
  config; every epoch appends one JSONL record; every figure is a pure
  function of that JSONL, never of in-memory training state.

## AI-assistance disclosure

Implementation and drafting in this project were carried out with the
assistance of an AI coding agent, operating under the frozen specification
above, preregistered predictions committed before each result, and an
automated test suite required to pass before and after every change. The
author designed the specification, every preregistered threshold, and the
overall research questions; reviewed every reported number against its
source log; and is responsible for all content, having independently
verified the theorem statements and proofs. This mirrors the disclosure in
the paper's Methods section (§3.4) — no AI tool is credited as an author.

## Authorship and how to cite

This work is licensed under **[CC BY 4.0](LICENSE)** — you may share and
adapt it for any purpose, including commercially, provided you give
appropriate credit, link to the license, and do not represent it, in whole
or in part, as your own original research. See [`CITATION.cff`](CITATION.cff)
for the canonical citation (also surfaced by GitHub's "Cite this
repository" button).

**G Karthik Koundinya** · Department of Computer Science and Engineering,
Geethanjali College of Engineering and Technology · ORCID:
[0009-0003-2088-6208](https://orcid.org/0009-0003-2088-6208) ·
karthikofficialmain@gmail.com

## References

Papyan, Han, Donoho (2020), *PNAS* 117(40):24652–24663 (classical Neural
Collapse). Du, Yang, Tao, Hsieh (2023), *Physical Review Letters* 131,
140601 (quantum Neural Collapse at the static optimum). San Sebastian,
Canizo, Orus (2026), arXiv:2602.08485 (empirical quantum NC study). Full
reference list: [`docs/references/REFERENCES.md`](docs/references/REFERENCES.md).
