# REANALYSIS.md -- Stage 7: normalized-witness reanalysis

Implements PLAYBOOK2.md Stage 7 / SPEC2_ADDENDUM.md section 11. No new
training runs. See CLAUDE.md for mandatory citations (Papyan/Han/Donoho 2020;
Du/Yang/Tao/Hsieh 2023; San Sebastian/Canizo/Orus 2026) plus the Phase 2
additions (Larocca et al. 2023; Bowles et al. 2024; Han/Papyan/Donoho 2022).

## Method

- `qnc1_fisher` (SPEC2 section 11.1) recomputed for every Phase 1 run
  directly from already-logged `qnc1_hs` and `class_mean_norms` fields --
  no checkpoint reload needed. Note: this is algebraically identical to the
  `qnc1_rel` field already present in every Phase 1 JSONL line
  (`qnc1_rel = qnc1_hs / mean_c ||M_c||^2_HS`, the same S_W/S_B ratio);
  Stage 7 gives that existing quantity its SPEC2 name and headline role.
- `qnc1_ratio`, `overlap_offdiag`, `purity_mean` (SPEC2 sections 11.2-11.3)
  recomputed by reloading checkpoint `state_dict`s (5-seed C=2 anchor,
  `configs/stage2_vqc_c2_diagnostic.yaml`) or reconstructing the seeded
  pre-training state (10-seed random-init null,
  `configs/stage5_random_init.yaml`, which trains 0 epochs and saves no
  checkpoints) and regenerating states via `VQC.state()`. No optimizer step
  is ever taken.

## Four-witness trained-vs-null comparison

| metric | trained (C=2 anchor, final epoch, n=5) | null (random-init, n=10) | Mann-Whitney U p-value |
|---|---|---|---|
| qnc1_trace | 0.945805 +/- 0.00593142 | 0.946773 +/- 0.00629237 | 0.6787 |
| qnc1_fisher | 47.6565 +/- 7.7181 | 49.7667 +/- 8.16969 | 0.5941 |
| qnc1_ratio | 1.36352 +/- 0.0514688 | 1.37748 +/- 0.0667624 | 0.4396 |
| purity_mean | 0.0557393 +/- 0.0064114 | 0.0546314 +/- 0.00666157 | 0.6787 |

## Verdict

No normalized witness separates trained from untrained states (all p >= 0.05) at this C=2 anchor; the Phase 1 null result is not a metric artifact at this configuration.
