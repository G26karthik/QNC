# REVIEW_CHECKLIST.md — Author sign-off before submission

`paper/main.tex` is a complete draft, compiled to `paper/main.pdf` (9
pages + appendix, MiKTeX pdflatex + bibtex, 4-pass build, zero errors,
only benign float-placement warnings). Nothing below is optional before
this becomes a submittable draft.

## 1. Theorem sign-offs (five items, all currently marked pending in the source)

Each is wrapped in a `% TODO-VERIFY-BY-AUTHOR ... % END TODO-VERIFY-BY-AUTHOR`
comment block in `paper/main.tex`'s Appendix B (search for
`TODO-VERIFY-BY-AUTHOR` to find all five). None may be treated as
established until you have independently checked the proof sketch (not
just the empirical corroboration, which is already machine-verified).

- [ ] **T1** (Theorem~1, "No loss-selected pressure along fibers") —
  chain-rule argument that $\nabla_\theta\mathcal{L}$ vanishes off the
  measured span. Short, should be quick to check.
- [ ] **T5** (Theorem~2, "Collapse by rotation", the centerpiece) —
  unitary-conjugation Hilbert-Schmidt isometry argument, plus its two
  corollaries (purity invariance; Du-overlap invariance and the
  training-unreachability claim about Du et al.'s optimum) and the
  entanglement-entropy non-invariance remark. This is the paper's central
  claim — check it most carefully, including the exact architectural scope
  (no re-uploading, fixed measurement) stated in `THEORY_NOTES.md`.
- [ ] **T2'** (Theorem~3, "ETF directions without box saturation") — the
  box-vertex cosine algebra is machine-checked (`tests/test_metrics.py`);
  the scale-absorption argument (why $\beta$ rather than the state's own
  norm grows) is informal and worth tightening.
- [ ] **T3** (Theorem~4, dimension counting) — direct dimension count under
  a stated isotropic-null assumption; check whether that assumption needs
  qualification.
- [ ] **T6** (Theorem~5, NC1$_z$/pseudoinverse instability) — perturbation
  argument for pseudoinverses near a rank change; check the claimed
  connection to standard pinv perturbation theory.

## 2. Fifteen-number audit

- [ ] Open `NUMBER_AUDIT.md` and spot-check at least 5 of the 15 sampled
  claims against the quoted source lines yourself (all 15 were checked by
  the drafting agent, but independent verification of a subset is worth
  the ten minutes).
- [ ] Confirm the ledger tally (19 CONFIRMED / 6 REFUTED / 4 PARTIAL / 1
  factual, of 30) matches Appendix A's table by counting it yourself once.

## 3. Abstract read-aloud check

- [ ] Read the abstract aloud once, unedited. Confirm: (a) it is under 250
  words (current draft: check word count before submission — count
  fluctuates with any edits), (b) the five claims appear in order and
  claim 2 is in its corrected radius/saturation form (not the earlier
  "ETF beats box" framing), (c) no banned word (novel, groundbreaking,
  paradigm, revolutionize, remarkably) appears anywhere in the abstract or
  body — a fresh grep is worth running after any edit:
  `grep -iE "novel|groundbreaking|paradigm|revolutioni[sz]e|remarkably" paper/main.tex`

## 4. Figure caption accuracy pass

- [ ] All 12 figures (`paper/figures/fig01`–`fig12`) are copied verbatim
  from verified `results/` paths (see `PAPER_SKELETON.md` and
  `RESULTS_V3.md` for the source path of each). Confirm each caption
  matches what the figure actually shows — captions were written from the
  findings-file descriptions, not by looking at rendered pixels; a visual
  check against `paper/main.pdf` is worth doing once.
- [ ] Confirm Figure 6 (`fig06_total_variance_rotation.png`, the rotation
  mechanism's own figure) and Figure 2 (`fig02_fiber_dip_diagnostic.png`,
  the measurement-subspace-discovery figure) — the paper's two
  load-bearing figures per `PAPER_SKELETON.md`'s note — render at
  legible resolution in the compiled PDF.

## 5. Author / affiliation fields

- [ ] `paper/main.tex`'s `\author{}` block currently contains three
  placeholder strings (`AUTHOR NAME (PLACEHOLDER -- FILL IN)`,
  `AFFILIATION (PLACEHOLDER -- FILL IN)`, `EMAIL (PLACEHOLDER -- FILL IN)`).
  All three must be replaced before submission — grep for `PLACEHOLDER` in
  `paper/main.tex` to find them.
- [ ] Confirm the Acknowledgments section's AI-assistance disclosure
  wording is acceptable to you and to any venue-specific disclosure policy
  you submit under (wording is currently generic; some venues require a
  specific disclosure format).

## 6. arXiv category and license

- [ ] **Primary category**: `quant-ph` (recommended, given the paper's
  primary contribution is a quantum-training-dynamics result).
- [ ] **Cross-list**: `cs.LG` (recommended, given the neural-collapse /
  machine-learning framing and audience).
- [ ] **License**: your decision — arXiv offers several (arXiv's own
  non-exclusive license, CC BY 4.0, CC BY-SA 4.0, CC BY-NC-SA 4.0, CC0).
  Not set anywhere in this draft; choose before submission.

## Build notes (for whoever compiles this next)

- Toolchain used: MiKTeX pdflatex + bibtex, 4-pass build
  (`pdflatex` → `bibtex` → `pdflatex` → `pdflatex`), zero errors, only
  benign `[h] -> [ht]` float-placement warnings.
- `paper/main.bbl` is committed alongside `paper/main.tex` so the PDF can
  be rebuilt with `pdflatex main.tex` alone (no bibtex rerun needed) unless
  `references.bib` changes, in which case rerun the full 4-pass build.
- One build issue was hit and fixed during drafting: square brackets in
  the `\author{}` placeholder text (`[AUTHOR NAME]`) broke hyperref's PDF
  metadata extraction ("Missing number, treated as zero"). Fixed by using
  bracket-free placeholder text. If you edit the author block, avoid
  literal `[`/`]` characters in it.
