# REFERENCES.md — Citation verification (Stage 17b, Task D)

Every citation used in RESULTS_V3.md / THEORY_NOTES.md / PAPER_SKELETON.md,
verified against the arXiv abstract page or publisher page (fetched during
this assembly pass, 2026-07-17). No citation below is guessed from memory
without a fetched source; where a title/author list could not be verified
it is flagged rather than asserted.

| Citation as used in this repo | Verified title | Verified full author list | Verified venue | Source |
|---|---|---|---|---|
| Papyan, Han, Donoho (2020), PNAS | Prevalence of Neural Collapse during the terminal phase of deep learning training | Vardan Papyan, X. Y. Han, David L. Donoho | PNAS 117(40):24652–24663 (2020) | [arXiv:2008.08186](https://arxiv.org/abs/2008.08186), [PNAS](https://www.pnas.org/doi/10.1073/pnas.2015509117) |
| Du, Yang, Tao, Hsieh (2023), PRL 131, 140601 | Problem-Dependent Power of Quantum Neural Networks on Multiclass Classification | Yuxuan Du, Yibo Yang, Dacheng Tao, Min-Hsiu Hsieh | Physical Review Letters 131, 140601 (2023) | [arXiv:2301.01597](https://arxiv.org/abs/2301.01597) |
| San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 | Empirical Study of Observable Sets in Multiclass Quantum Classification | Paul San Sebastian, Mikel Canizo, Roman Orus | arXiv preprint, 2026 | [arXiv:2602.08485](https://arxiv.org/abs/2602.08485) |
| Larocca, Ju, Garcia-Martin, Coles, Cerezo (2023), Nat. Comput. Sci. | Theory of overparametrization in quantum neural networks | Martin Larocca, Nathan Ju, Diego Garcia-Martin, Patrick J. Coles, M. Cerezo | Nature Computational Science 3, 542–551 (2023) | [Nature](https://www.nature.com/articles/s43588-023-00467-6), [PubMed](https://pubmed.ncbi.nlm.nih.gov/38177434/) |
| Bowles, Ahmed, Schuld (2024), arXiv:2403.07059 | Better than classical? The subtle art of benchmarking quantum machine learning models | Joseph Bowles, Shahnawaz Ahmed, Maria Schuld | arXiv preprint, 2024 | [arXiv:2403.07059](https://arxiv.org/abs/2403.07059) |
| Han, Papyan, Donoho (2022), ICLR | Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path | X. Y. Han, Vardan Papyan, David Donoho | ICLR 2022 (Outstanding Paper Award) | [arXiv:2106.02073](https://arxiv.org/abs/2106.02073) |
| Mixon et al. (UFM gradient-flow collapse) | Neural collapse with unconstrained features | Dustin G. Mixon, Hans Parshall, Jianzong Pi | Sampling Theory, Signal Processing, and Data Analysis 20(2) (2022) | [arXiv:2011.11619](https://arxiv.org/pdf/2011.11619) |
| Yang et al. (NeurIPS 2022, fixed-ETF-classifier collapse) | Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network? | Yibo Yang, Shixiang Chen, Xiangtai Li, Liang Xie, Zhouchen Lin, Dacheng Tao | NeurIPS 2022 | [OpenReview](https://openreview.net/pdf?id=y5W8tpojhtJ), [GitHub](https://github.com/NeuralCollapseApplications/ImbalancedLearning) |
| QMT paper, arXiv:2606.22551 | Mitigating Measurement-Induced Training Instability in Hybrid Quantum Neural Networks for Protein Classification | Milton Mondal, Sushovan Chanda, Mohamad Mahdi Alawieh, Brijesh Sukhadiya, Donatus Krah, Clinton Gonsalves, Antonios Ntolkeras, Silvio O. Rizzoli, Ali H. Shaib | arXiv preprint, 2026 | [arXiv:2606.22551](https://arxiv.org/abs/2606.22551) |

## Flags

None of the 9 citations above required flagging — every author list was
confirmed against a fetched arXiv abstract page or publisher page (PNAS,
Nature Computational Science, or the paper's own arXiv listing).

Note two small spelling/formatting corrections surfaced during
verification, propagated wherever these names are cited in RESULTS_V3.md /
THEORY_NOTES.md / PAPER_SKELETON.md:
- "San Sebastian, Canizo, Orus" — middle author's name is "Canizo" per the
  arXiv listing (no diacritic rendered in the source metadata); kept as
  written elsewhere in this repo for consistency, flagged here only in case
  the canonical spelling is "Cañizo."
- Larocca et al.'s third author is "Garcia-Martin" (no diacritic in the
  fetched metadata); same note applies.
