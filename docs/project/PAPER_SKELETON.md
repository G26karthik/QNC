# PAPER_SKELETON.md — Section-by-section outline for the final paper

Target format: 9 pages main text + appendix, single-column-equivalent
budget (approximate word counts assuming a standard two-column ML venue
template, ~250 words/half-column). Every figure listed below is verified to
exist on disk as of this assembly pass (Stage 17).

## Title candidates

1. **"Collapse by Rotation: Training-Dynamics Neural Collapse in Variational
   Quantum Circuits"** (lead candidate)
2. "Collapse by Rotation: How Cross-Entropy Redistributes Variance in
   Quantum Classifiers Without Shrinking It"
3. "Measurement-Subspace Collapse: Quantum Neural Collapse Is a Rotation,
   Not a Contraction"

## Abstract — five claims, in order

1. Full Hilbert-space collapse witnesses (trace distance, Fisher ratio,
   purity, ETF geometry) remain statistically indistinguishable from an
   untrained null throughout training and across an overparameterization
   transition (Phase 1/2 null, RESULTS_V2.md §2, §4) — reconciled, not
   contradicted, by measurement-subspace collapse: cross-entropy only reads
   the C-dimensional projection z, and collapse there is real and large
   (NC1_z decays 2–3 orders of magnitude, Stage 12 P1).
2. The measured-subspace class-mean geometry converges to simplex-ETF
   directions (d_etf 0.072) under a learnable logit scale (beta) without
   ever saturating at the box-vertex configuration's fixed radius (final
   class-mean radius ~20% of the box-saturation value) — at C=3 the two
   models share the same angular target, so directions alone cannot
   distinguish them; the discriminator is that beta, not the state's own
   norm, absorbs CE's margin pressure (Stage 12 P3, THEORY_NOTES T2′).
3. Without data re-uploading, this collapse is provably (and numerically,
   to machine precision) a rigid rotation of the state constellation, not a
   contraction: total Hilbert-Schmidt within-class variance is exactly
   conserved (Stage 14B HR2, ~1e-15 relative change) while the loss-visible
   NC1_z diagnostic still decays by more than an order of magnitude
   (Stage 14B HR3).
4. This rotation-by-unitary mechanism and its measured-subspace/fiber
   dichotomy replicate at a second qubit count (n=6) and strengthen with n
   exactly as dimension counting predicts (Stage 15 HN1–HN3).
5. The mechanism is validated, within its licensed scope (agreement/
   disagreement within shot noise, nothing stronger), on real IBM quantum
   hardware: the pipeline measures a genuine trained-vs-untrained effect
   (94.2x, Stage 16b HW3), and hardware disagreement with simulation is
   well described by a single scalar contraction with angles preserved
   (Stage 16b HW1/HW2).

## Section outline

| # | Section | Figure | Budget |
|---|---|---|---|
| 1 | Introduction & positioning (Du et al. proved the static optimum; we study training dynamics, geometry, entanglement's role — SPEC.md §0.3) | none | 0.75 page |
| 2 | Related work (Du et al., Larocca et al., Bowles et al., San Sebastian et al., Han et al., Mixon et al., Yang et al., QMT/Mondal et al.) | none | 0.5 page |
| 3 | Model & methods (angle encoding, SEL ansatz, TPT definition, z-space witnesses SPEC3 §18) | none | 0.75 page |
| 4 | Phase 1/2: the full-space null and the capacity transition | `results/sweeps/transition_L_order_parameter.png` | 1 page |
| 5 | Measurement-subspace collapse discovery (Stage 12: P1–P5) | `results/transition_L_L8_aggregate/figures/fiber_dip_diagnostic.png` | 1 page |
| 6 | ETF vs box-vertex geometry (Stage 12 P3, theory T2′) | `results/transition_L_L8_aggregate/figures/etf_formation.png` | 0.5 page |
| 7 | Measurement-expressivity mechanism (Stage 13/13B: H1–H3, depth-confound correction) | `results/sweeps/stage13_expressivity_aggregate/expressivity.png` | 0.75 page |
| 8 | Robustness arms (Stage 14: arms 1–6, ruling out beta-escape-valve and label-smoothing mechanisms) | `results/stage14_long_horizon_aggregate/figures/trace_w_drift.png` | 0.5 page |
| 9 | Collapse by rotation — the mechanism (Stage 14B: HR1–HR5, theory T5 centerpiece) | `results/stage14b_reanalysis/figures/total_variance_rotation.png` | 1.25 page |
| 10 | n=6 replication (Stage 15: HN1–HN3, theory T3 dimension counting) | `results/stage15_n6_reanalysis/figures/n4_vs_n6_dichotomy.png` | 0.75 page |
| 11 | Hardware validation (Stage 16/16b: HW1–HW3, theory T6 pinv caution) | `results/stage16_hardware/d9cf942neu4c739lt300/figures/zspace_hw_rescaled_vs_sim.png` | 1 page |
| 12 | Discussion, limitations, future work (RESULTS_V3.md §(f)) | none | 0.5 page |
| — | **Total main text** | | **~9 pages** |
| A | Appendix: full preregistration ledger (RESULTS_V3.md §(b)) | none | unlimited |
| B | Appendix: theory notes T1, T2′, T3, T5, T6 in full (THEORY_NOTES.md) | none | unlimited |
| C | Appendix: additional figures (`results/sweeps/stage10_ablation_20260713-115624/ablation_bounds.png`, `results/stage7_reanalysis/figures/four_witness_comparison.png`, `results/transition_L_L8_aggregate/figures/nc1z_decomposition.png`, `results/stage16_hardware/d9cf942neu4c739lt300/figures/nc1z_hw_vs_sim.png`) + full run_id traceability | see list | unlimited |

## Notes

- Section 9 (rotation mechanism) and section 5 (measurement-subspace
  discovery) are the two load-bearing sections; if page budget must be cut,
  compress section 7 (expressivity — H1/H2 are both PARTIAL, supporting
  detail rather than a headline claim) or section 8 (robustness — mostly
  ruling out alternative mechanisms) first.
- All figure paths above verified to exist on disk during this Stage 17
  assembly pass (2026-07-17).
