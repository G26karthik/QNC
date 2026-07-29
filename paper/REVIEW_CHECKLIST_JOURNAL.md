# REVIEW_CHECKLIST_JOURNAL.md — pre-submission checklist (Quantum Machine Intelligence draft)

## Title-page completeness

- [x] Title: "Collapse by Rotation: Training-Dynamics Neural Collapse in
  Variational Quantum Circuits, Verified on Real Vision Data" — combines
  `paper/main.tex`'s title with `paper_icvgip/main.tex`'s real-data
  addition.
- [x] Author: "G Karthik Koundinya" (real name, `\author*[1]{\fnm{G
  Karthik} \sur{Koundinya}}`) — no "Anonymous Author(s)" placeholder.
- [x] Affiliation: "Department of Computer Science and Engineering,
  Geethanjali College of Engineering and Technology" (`\affil*[1]`),
  country "India" added as the only additional address field (city/street/
  postcode were not supplied by the user and were not guessed).
- [x] Corresponding email: `karthikofficialmain@gmail.com`, rendered as a
  clickable `mailto:` link under the title.
- [x] ORCID `0009-0003-2088-6208` — the `sn-jnl` class has no dedicated
  ORCID macro, so this is rendered as a plain labeled line ("ORCID: G
  Karthik Koundinya, 0009-0003-2088-6208", hyperlinked to
  `orcid.org/...`) directly under the title block, page 2. Confirmed
  present in the compiled PDF (page 2 of 31), not just the source.
- [x] Grep across `main.tex` and `pdftotext` extraction of `main.pdf` for
  "anonymous" and "double-blind": zero matches in either. No double-blind
  artifacts of any kind remain (the ICVGIP draft's `review=true,
  anonymous=true` acmart options are absent — this uses the Springer
  Nature `sn-jnl` class instead, which is not a blind-review template).

## AI disclosure

- [x] Present, not deferred: Section 3.4 ("AI-assistance disclosure"),
  page 6, a full paragraph in the Methods section (not the Acknowledgments
  or a footnote).
- [x] Accurate against what was actually done this session and across the
  project's history: states the AI agent operated under a frozen
  specification, preregistered predictions committed before each result,
  and an automated test suite (173 tests, matching `CLAUDE.md`'s stated
  invariant); states the agent's role (experiment orchestration,
  metrics-code implementation, drafting) and the author's role (research
  design, threshold-setting, number verification, proof verification);
  states no AI tool is credited as an author. This mirrors
  `paper/main.tex`'s existing (shorter) Acknowledgments-section disclosure,
  expanded per the task's requirement to state it explicitly in Methods
  rather than defer it.
- [x] Author Contribution subsection of Declarations cross-references
  Section 3.4 rather than duplicating or contradicting it.

## Statements and Declarations

- [x] Section present, titled "Declarations", placed in the backmatter
  before the reference list (via `\backmatter`), per the `sn-jnl` template's
  own convention.
- [x] **Funding** — verbatim: "The authors did not receive support from
  any organization for the submitted work."
- [x] **Competing Interests** — verbatim: "The authors have no relevant
  financial or non-financial interests to disclose."
- [x] **Data Availability** — describes code, configs, preregistrations,
  per-epoch logs, and figures as available in the project repository. The
  `[REPOSITORY URL]` placeholder has been **filled in**: the repository is
  public at `https://github.com/G26karthik/QNC` and the URL is rendered
  in the compiled PDF (confirmed via `pdftotext` extraction, not just the
  source). This item is now resolved, not open.
- [x] Additional standard `sn-jnl` Declarations subsections not explicitly
  requested but included for template completeness, all "Not applicable"
  where accurate (Ethics approval/consent to participate, Consent for
  publication, Materials Availability) or a same-repository pointer (Code
  Availability), plus a single-author Author Contribution statement.

## Abstract word count

- [x] **227 words** (counted programmatically from the `\abstract{...}`
  source, stripping LaTeX markup), within Springer's stated 150–250 word
  range with margin on both sides.
- [x] No undefined abbreviations: NC, VQC, and $C$ are all defined at
  first use within the abstract itself.

## Keywords

- [x] Five keywords: "neural collapse, variational quantum circuits,
  representation geometry, measurement subspace, terminal phase of
  training" — within the requested 4–6 range (tighter than ICVGIP's looser
  count, per the task's requirement).

## Citation style consistency

- [x] `\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}` — Springer Nature's
  Math and Physical Sciences **Author-Year** reference style, compiled with
  `sn-mathphys-ay.bst` (downloaded from the template's own distribution,
  not improvised).
- [x] All in-text citations use `\citet{}`/`\citep{}` (natbib, loaded by
  the class), rendering as "Papyan et al. (2020)" / "(Papyan et al., 2020)"
  throughout — spot-checked across Introduction, Related Work, Theory, and
  every Results section during the page-by-page visual review.
- [x] Reference list (page 24–26 of the compiled PDF) is alphabetized by
  first-author surname (Abbas → Zhu), consistent with the author-year
  style; no numbered brackets anywhere.
- [x] Zero `natbib` "undefined citation" warnings in the final compile log
  (`pdflatex` → `bibtex` → `pdflatex` ×2, all four passes exit 0; verified
  via `grep -i undefined` on the final pass's log).

## Theorem proofs: complete, not deferred

- [x] Section 4 ("Theory: Collapse Is a Rotation") and its subsection 4.1
  contain **full, complete proofs** for all seven theorem-environment
  statements (Lemma 1, Corollary 2, Proposition 3 [3-step proof],
  Corollary 4, Remark 1, Proposition 5, Proposition 6, Remark 2) — every
  `\begin{proof}...\end{proof}` block is a complete argument, not a
  "proof sketch (to be verified by author)" placeholder. This is a
  deliberate change from `paper/main.tex`'s appendix, which wrapped
  visible "(to be verified by author)" text into several proof captions
  and internal `TODO-VERIFY-BY-AUTHOR` LaTeX comments (invisible in the
  PDF but present in the source) around every theorem; this journal draft
  instead reuses `paper_icvgip/main.tex`'s already-complete proof text
  (which itself resolved those TODOs into finished derivations), consistent
  with the AI-disclosure statement (§3.4) that the author has verified the
  theorem statements and proofs.
- [x] No `TODO-VERIFY-BY-AUTHOR` markers, visible or as LaTeX comments,
  remain anywhere in `paper_journal/main.tex` (grepped, zero matches).
- [ ] **This checklist item records that the proofs are textually complete
  and self-consistent, not that a domain expert outside this project has
  independently re-derived them from first principles.** The author's own
  verification is asserted in Section 3.4 and the Author Contribution
  statement; this remains the author's responsibility to stand behind at
  submission, as with any paper.

## QTML2025 topical collection

- [x] This draft is **not** tagged to, and makes no reference to, the
  "Quantum and Quantum-inspired Machine Learning: from foundations to
  theoretical advances" (QTML2025) topical collection — consistent with
  prior research establishing this paper targets the general Research
  Article submission track, not that collection.
- [ ] **OPEN ITEM FOR THE USER:** if Springer's online submission portal
  for *Quantum Machine Intelligence* offers the QTML2025 topical collection
  as a selectable option during upload, whether to select it is the user's
  choice to make at submission time — nothing in this draft or repository
  should be read as having decided that question.

## Compilation and rendering

- [x] Real Springer Nature LaTeX template used: `sn-jnl.cls` and
  `sn-mathphys-ay.bst`, fetched from the template's own GitHub mirror
  (`DanySK/template-latex-springer-nature-sn-jnl`, itself a packaging of
  Springer Nature's official author template) after confirming the class
  is not available pre-installed via MiKTeX's package repository or CTAN
  under a directly fetchable name — not the plain-`article` fallback.
- [x] Compiled cleanly: `pdflatex` → `bibtex` → `pdflatex` ×2, all four
  passes exit code 0, zero LaTeX errors, zero undefined citations or
  references on the final pass.
- [x] **31 pages**, no page-count constraint applied anywhere in the
  source (no `\documentclass` page-limit option, no compressed proofs, no
  "omitted for space" language) — the fullest accurate version of the
  paper, per the task's explicit instruction.
- [x] Every one of the 31 pages rendered to PNG (`pdftoppm -r 150`) and
  visually inspected this session, including title/abstract, every
  Introduction/Related Work/Methods page, every theorem and proof, every
  figure, every table (including the two multi-page `longtable` ledger
  spans), the Declarations section, the full reference list, and both
  appendices.
- [x] One real rendering bug was found and fixed during review: the
  preregistration ledger's `longtable` initially rendered "CONFIRMED" and
  "UNSCOREABLE" verdicts running directly into the adjacent column's text
  with zero visible gap (the words were wider than their 1.6–1.9cm column
  and overflowed past the inter-column `\hspace`, rather than the gap being
  literally absent) — fixed by widening the Verdict column and switching
  the whole table to `\footnotesize`; re-rendered and re-inspected at high
  resolution to confirm the fix.
- [x] `pytest` untouched: no files under `src/`, `tests/`, or `configs/`
  were modified in this task; only `paper_journal/` was created.

## Scope note

This draft's Appendix A extends the leg-split preregistration count
established in `STAGE20_FINDINGS.md` (which itself only recounts Stages
12–19) through this paper's own Stage 20 arms, arriving at a new headline
figure (52 items, deliverables excluded) not previously stated anywhere in
the repository. The full arithmetic is shown inline in `main.tex`'s
Appendix A and traced in `NUMBER_AUDIT_JOURNAL.md` row 15 — this is the
single highest-priority item for the author to independently re-verify
before submission, precisely because it is new arithmetic rather than a
copied number.
