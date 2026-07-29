# STAGE19_FINDINGS.md — Vision datasets + comparative study (ICVGIP 2026)

Citations mandatory: Papyan, Han, Donoho (2020), PNAS (classical NC); Du,
Yang, Tao, Hsieh (2023), PRL 131, 140601 (quantum NC at the static optimum);
San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 (empirical QNC study).

Scored against PREDICTIONS.md's "Stage 19 preregistration" section (Arms
V1/V2/V3). All 26 VQC runs and 6 classical MLP runs referenced below
completed to their full epoch budget (verified per-run against
`metrics.jsonl`); none were left partially trained.

## Status

| Arm | Status |
|---|---|
| V1(i) reupload=false total_var invariance | **CONFIRMED**, 5/5 seeds, both datasets |
| V1(ii) NC1_z decay in TPT-reaching seeds | **UNSCOREABLE** — 0/20 TPT-reaching seeds in Arm V1 (precondition never met) |
| V1(iii) TPT fractions | reported factually below |
| V2(a) classical MLP | **CONFIRMED** (feature-space collapse); full-space leg of the prediction is not well-posed for a fixed input (see below) |
| V2(b) VQC no-reupload | **CONFIRMED** (full-space invariance); z-space collapse direction holds but magnitude only informally checked (no TPT gate available) |
| V2(c) VQC reupload | **CONFIRMED** (both legs) |
| V2(d) VQC R3 (trainable measurement) | **REFUTED** on z-space direction (NC1_z increased, not decreased, in 5/6 seeds); full-space leg CONFIRMED |
| V3 scaling figure | done, `figs/fig18_stage19_scaling.png` |

## Preregistration deviations (flagged, not silently applied)

1. **L=8 → L=16 for all VQC arms, both datasets.** The written trim rule
   (PREDICTIONS.md) only covered a Fashion-MNIST L=8 TPT-reachability
   probe. MNIST L=8 R0 was undetected in advance and also failed to reach
   TPT (train acc plateau ~0.90–0.93 over 600 epochs, seed 0, both the
   `stage19_mnist_campaign` and `stage19_r3_mnist` configs). Fashion-MNIST's
   own L=8 probe (2 seeds) also failed 0/2. L=16 was applied to both
   datasets for consistency; a 2-seed L=32 capacity-check probe was run in
   parallel and showed the same plateau, so L=16 was kept as the reported
   depth rather than doubling again indefinitely.
2. **Arm V2(b)/(c)/(d) run at L=16, not the originally planned L=8**, for
   the same reason (they reuse the depth-16 base configs above rather than
   a separate L=8 config).
3. **Arm V2(b)/(c) report 5 seeds, not 3.** These reuse the full Arm V1
   campaign seeds (0–4) rather than a separate 3-seed run, per
   PREDICTIONS.md's own note that "seeds 0-2 of each variant are reused by
   Arm V2(b)/(c)." All 5 available seeds are reported rather than
   artificially dropping 2.
4. **Infrastructure**: the background-task execution environment killed
   long-running training sweeps at a wall-clock ceiling of roughly
   20–40 minutes, independent of sandboxing settings; all runs reported
   here were ultimately completed via detached OS processes (explicit user
   authorization) and verified to reach their full target epoch count
   (600) directly from each run's `metrics.jsonl`, not inferred from
   process exit status.

## Arm V1 — real vision datasets, full witness set

**(i) reupload=false total_var invariance** (`total_var_max_rel_change`,
5 seeds each):

| Dataset | Seeds | Max relative change |
|---|---|---|
| MNIST {0,1,2} | 0–4 | 4.5e-15 to 8.3e-15 |
| Fashion-MNIST {0,1,2} | 0–4 | 4.7e-15 to 9.5e-15 |

**CONFIRMED**, 5/5 seeds both datasets, ~10 orders of magnitude below the
1e-6 threshold — the rotation theorem (T5) holds on real image data
exactly as it does on the synthetic blobs3 recipe, confirming it is a
property of the no-reupload encoding structure, not of the training data.

**(ii) NC1_z decay in TPT-reaching seeds**: **UNSCOREABLE**. 0/20 seeds
across the MNIST and Fashion-MNIST campaigns (both reupload arms) reached
TPT (see (iii)), so there is no seed set to which the preregistered
criterion applies. Reported factually, ungated by TPT: NC1_z final/init
ratios still show substantial decay in every seed regardless —
reupload_true: 0.0006–0.013 (MNIST/FMNIST combined); reupload_false:
0.010–0.19. This is a positive but *unregistered* secondary observation:
the z-space collapse mechanism operates well before perfect training
accuracy, but the preregistered claim specifically about TPT-reaching
seeds cannot be evaluated here.

**(iii) TPT fractions** (factual):

| Dataset | reupload=true | reupload=false |
|---|---|---|
| MNIST {0,1,2}, L=16, R0 | 0/5 | 0/5 |
| Fashion-MNIST {0,1,2}, L=16, R0 | 0/5 | 0/5 |
| MNIST, R3, L=16 | 0/3 | — |
| Fashion-MNIST, R3, L=16 | 0/3 | — |

0/26 VQC runs in this stage reached TPT, at L=8, L=16, or (2-seed probe)
L=32. Over the same PCA-4 features, the capacity-matched classical MLP
reaches TPT in 5/6 seeds (see Arm V2). This points to a genuine
ansatz-expressivity/optimization gap on real image data — not
undertraining — rather than any bug in the pipeline (train accuracy
plateaus stably at 0.87–0.96 for hundreds of epochs past the point where
it would need to move to reach TPT).

## Arm V2 — comparative study on identical PCA-4 features

| Approach | Seeds (n) | Test acc (mean) | TPT | z-space NC1 ratio (final/init, mean) | Full-space classification |
|---|---|---|---|---|---|
| (a) Classical MLP | 6 | 0.935 | 5/6 | 0.515 (feature space, 16-d penultimate) | N/A — see note below |
| (b) VQC no-reupload, R0 | 10 | 0.928 | 0/10 | 0.060 | **invariant** (rel. change ~1e-14–1e-15) |
| (c) VQC reupload, R0 | 10 | 0.713 | 0/10 | 0.0046 | **mildly contracting** (mean 2.8%) |
| (d) VQC R3 (trainable measurement) | 6 | 0.713 | 0/6 | 1.92 (**increase**) | **mildly contracting** (mean 3.4%) |

Per-approach scoring against PREDICTIONS.md's directional predictions:

- **(a) classical MLP** — predicted to collapse in both feature space and
  full input space. Feature-space collapse is **CONFIRMED**: NC1 ratio
  <1 in 6/6 seeds (range 0.31–0.91), TPT reached in 5/6. The "full input
  space" leg of the prediction is **not well-posed as stated**: the raw
  PCA-4 input to a classical MLP is fixed data, untouched by training, so
  it cannot show training-induced collapse or contraction by
  construction — unlike the VQC's full Hilbert-space representation,
  whose invariance is a nontrivial consequence of a *fixed but nonlinear*
  quantum encoding (T5), a classical linear input has no analogous
  structure to test. This is a scoping gap discovered during scoring, not
  a result.
- **(b) VQC no-reupload** — predicted z-space collapse only, full-space
  exactly invariant. Full-space invariance is **CONFIRMED** at machine
  precision (~1e-14–1e-15), the cleanest result in this stage and an exact
  match to the (i) result above (same runs). z-space collapse direction
  holds in 10/10 seeds (ratio <1) but the informal magnitude is milder
  than the reupload arm and cannot be checked against the TPT-gated
  threshold (0/10 TPT).
- **(c) VQC reupload** — predicted z-space collapse plus a small nonzero
  full-space contraction, mirroring HR1's 0.73% decline at n=4.
  **CONFIRMED**: z-space collapses far past 1 order of magnitude in
  10/10 seeds (ratio 0.0006–0.013) and full-space shows a small,
  consistent contraction (2.0–3.6%, mean 2.8%) — larger than HR1's
  original 0.73% but the same direction and order of magnitude. Notably,
  test accuracy is markedly *lower* than the no-reupload arm (0.713 vs
  0.928) despite the stronger z-space collapse — reupload hurts
  generalization on real image data even as it collapses the loss-visible
  subspace more aggressively.
- **(d) VQC R3 (trainable measurement)** — predicted to pattern with (b):
  z-space collapse plus small full-space contraction similar to (c). The
  full-space leg is **CONFIRMED** (mean 3.4% contraction, same order as
  (c)). The z-space leg is **REFUTED**: NC1_z *increased* (ratio >1) in
  5/6 seeds (0.57–3.67, mean 1.92), the opposite of every other arm in
  this stage. This is the most notable negative result in Stage 19 — at
  L=16 with 2 trainable measurement layers, the readout head appears to
  actively work against z-space collapse rather than mirror the
  no-reupload behavior, contradicting the preregistered mechanism
  argument that R3's extra parameters sit downstream of the same
  no-reupload state-preparation unitary as (b). Whether this is specific
  to L=16 (vs. the originally planned L=8) or to real image-data features
  is not disentangled here and would need a dedicated follow-up.

fiber_fraction (measurement-subspace fraction) across all 26 VQC runs
ranges 0.984–0.993, consistent with the existing n=4 reference value
(0.9934, Stage 12 P4) and the isotropic null — a useful cross-validation
of Arm V3's figure using entirely new (real image) data.

## Arm V3 — scaling figure

`figs/fig18_stage19_scaling.png`: fiber_fraction vs qubit count (n=4, 6, 8)
with the isotropic-null prediction curve 1 − 3/(4^n − 1) overlaid. Built
from already-reported values (Stage 12 P4, Stage 15 HN2, Stage 18 B3); not
independently scored, a visualization deliverable per the preregistration.

## Run inventory

All run_ids and per-run scored quantities (total_var_max_rel_change,
nc1_z_ratio, fiber_fraction_final) are in `stage19_score.json`, generated
by `stage19_score.py`. Classical MLP per-run numbers are in
`results/stage19_classical_{mnist,fmnist}_pca4_default_s{0,1,2}_*/metrics.jsonl`.
