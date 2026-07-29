# STAGE20_FINDINGS.md -- Real-data TPT attempt + contraction-vs-rotation contrast

Citations mandatory: Papyan, Han, Donoho (2020), PNAS (classical NC); Du,
Yang, Tao, Hsieh (2023), PRL 131, 140601 (quantum NC at the static optimum);
San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 (empirical QNC study).

Scored against PREDICTIONS.md's "Stage 20 preregistration" section (Arms
W1/W2/W3/W4).

## Status

| Arm | Status |
|---|---|
| W1(i) TPT fraction vs samples_per_class | **CONFIRMED** -- spc=50: 10/10 TPT; spc=100: 9/10 TPT (5/5 reupload=true, 4/5 reupload=false) |
| W1(ii) NC1_z decay in TPT-reaching seeds | **CONFIRMED**, both spc cells, both reupload arms (median ratio well below 0.1) |
| W1(iii) reupload=false total_var invariance | **CONFIRMED**, both spc cells, all 10 seeds (~1e-15, machine precision, regardless of TPT) |
| W2 classical tr(Sigma_W) vs VQC total_var | **REFUTED** (direction) -- classical tr(Sigma_W) GROWS ~27-163x, does not contract; VQC total_var invariance CONFIRMED at machine precision |
| W3 ledger recount + table split | done |
| W4 amplitude-encoding probe | **done** -- 4/4 TPT (both reupload arms); total_var invariance CONFIRMED under a second encoding family |

## Arm W1 -- real-data TPT attempt

**Major result: the gap Stage 19 left open is closed for samples_per_class=50.
This is the first real-data TPT in this project's vision-dataset work
(0/26 in Stage 19 at spc=200).**

Probe (spc=50, 2 seeds/arm) reached TPT in 4/4 seeds, triggering the
fixed full-campaign trim rule. Full spc=50 cells (5 seeds/arm, probe seeds
0-1 + remainder seeds 2-4) are now complete:

| Variant | Seeds | TPT | Test acc (final), all 5 | NC1_z ratio (final/init), all 5 | Median NC1_z ratio | total_var max-rel-change, all 5 |
|---|---|---|---|---|---|---|
| reupload=true | 5 | **5/5** | 0.633, 0.600, 0.533, 0.600, 0.467 | 0.000544, 0.002594, 0.003686, 0.000944, 0.001450 | 0.00145 | 0.0226, 0.0273, 0.0428, 0.0281, 0.0199 (mild contraction, expected for reupload=true, not gated by (iii)) |
| reupload=false | 5 | **5/5** | 0.967, 0.867, 0.900, 0.933, 0.967 | 0.030518, 0.109447, 0.027057, 0.024795, 0.003407 | 0.02706 | 4.30e-15, 6.56e-15, 5.90e-15, 8.83e-15, 6.48e-15 (machine precision, (iii)'s claim) |

spc=100 cells (5 seeds/arm, fresh seeds 0-4) are now also complete:

| Variant | Seeds | TPT | Test acc (final), all 5 | NC1_z ratio (final/init), all 5 | Median (TPT-reaching only) | total_var max-rel-change, all 5 |
|---|---|---|---|---|---|---|
| reupload=true | 5 | **5/5** | 0.700, 0.633, 0.817, 0.683, 0.683 | 0.001946, 0.001506, 0.003874, 0.003085, 0.001838 | 0.00195 | 0.0191, 0.0211, 0.0342, 0.0247, 0.0185 (mild contraction, not gated by (iii)) |
| reupload=false | 5 | **4/5** (seed 1 did not reach TPT) | 0.900, 0.933*, 0.983, 0.917, 0.967 (*seed 1 not TPT-reaching, test acc reported regardless) | 0.007341, 0.056466*, 0.029094, 0.045854, 0.003305 (*seed 1 not TPT-reaching, excluded from the gated median) | 0.01822 | 5.47e-15, 5.31e-15, 5.18e-15, 9.29e-15, 7.50e-15 -- invariant in ALL 5, including the non-TPT seed |

**Total TPT fraction: spc=50 10/10, spc=100 9/10.**

**Scoring:**
- **(i) CONFIRMED**: spc=50's TPT fraction >= spc=100's in both reupload
  arms, per the fixed scoring rule (reupload=true: 5/5 >= 5/5, tied at
  ceiling in both cells; reupload=false: 5/5 > 4/5, a real difference in
  the predicted direction). The effect is modest (90% vs 100%, not a sharp
  cliff) and only visible in the reupload=false arm -- reupload=true was
  already at ceiling (5/5) at both sample sizes, so it cannot show the
  predicted direction one way or the other. Reported factually: this is a
  real but small effect, not a dramatic sample-size cliff.
- **(ii) CONFIRMED**, both spc cells, both reupload arms: median NC1_z
  ratio over TPT-reaching seeds is 0.00145/0.00195 (reupload=true, spc=50/
  100) and 0.02706/0.01822 (reupload=false, spc=50/100) -- all far under
  the 0.1 threshold (roughly 2-3 orders of magnitude decay). Individual-
  seed variance is visible in the tables above (e.g. spc=50 reupload=false
  seed 1's 0.1094 slightly exceeds 0.1 on its own; the median-based
  verdict does not hide this).
- **(iii) CONFIRMED**, both spc cells, all 10 reupload=false seeds:
  total_var max relative change is 4.30e-15 to 9.29e-15 throughout, ~9
  orders of magnitude below the 1e-6 threshold. Critically, this holds
  identically in spc=100 seed 1, the ONE seed across both cells that did
  NOT reach TPT (5.31e-15, indistinguishable from the TPT-reaching seeds)
  -- direct, clean confirmation that T5's invariance is a structural
  property of the no-reupload encoding, independent of whether the
  terminal training regime is ever reached.

**Overall: Arm W1 is CONFIRMED on all three predictions.** This closes the
gap Stage 19 left open (0/26 real-data TPT at spc=200): dropping
samples_per_class to 50-100 is sufficient to reach the terminal phase of
training on real MNIST features, and every witness behaves exactly as it
does on synthetic data once TPT is reached.

## Arm W4 -- amplitude-encoding probe

**Result: also 4/4 TPT.** Every probe seed (both reupload arms) reached
TPT under amplitude encoding, on MNIST PCA-16 -> 4-qubit amplitude
embedding, spc=50, L=16, R0:

| Variant | Seeds | TPT | NC1_z ratio (final/init) | total_var max-rel-change | test acc (final) |
|---|---|---|---|---|---|
| reupload=true | 2 | **2/2** | 0.002276, 0.003328 | 0.01475, 0.01510 | **0.367, 0.367** (chance level, C=3) |
| reupload=false | 2 | **2/2** | 0.000870, 0.001106 | 4.98e-15, 5.23e-15 | 0.933, 0.867 |

No directional TPT prediction was registered (a scoping probe by design;
per PREDICTIONS.md, every outcome is reportable) -- 4/4 TPT is the "real
data terminal-phase point" outcome, and it arrived alongside W1's angle-
encoding TPT in the same session, so the angle-encoding capacity/manifold
diagnosis in PREDICTIONS.md's Arm W4 reasoning is no longer the operative
explanation for Stage 19's earlier 0/26: BOTH encodings interpolate once
samples_per_class drops to 50. This reframes the Stage 19 null as a
sample-size/overfitting-regime effect rather than an encoding-specific
one, on the evidence gathered so far.

**total_var invariance sub-claim: CONFIRMED under a second encoding
family.** reupload=false total_var max relative change is 4.98e-15 and
5.23e-15 -- machine precision, matching every angle-encoding result in
this codebase. T5's invariance argument only requires the encoding be
applied once and be parameter-independent; this is now verified
empirically under `qml.AmplitudeEmbedding`, not just `RY` angle encoding,
directly answering the "encoding-specific" objection at the theorem level
rather than by analogy.

**Notable secondary finding (not preregistered, reported factually):**
amplitude encoding with reupload=true reaches TPT (train accuracy 1.0) but
test accuracy sits AT CHANCE (0.367, C=3 chance=0.333) in both seeds --
severe memorization/overfitting, not generalization, despite the training-
side collapse signal (NC1_z ratio ~0.003) looking identical in kind to
every other TPT-reaching arm in this stage. reupload=false generalizes
normally (0.867-0.933). This is an unusually clean example of the same
training-side collapse metric co-occurring with opposite generalization
outcomes depending on reupload -- consistent with the reupload-hurts-
generalization pattern already noted in Stage 19 V2(c), now replicated
under a second encoding.

## Arm W2 -- contraction-vs-rotation contrast

**Result: the predicted direction is REFUTED, and the actual result is a
more dramatic version of the intended contrast, not a weaker one.**

Classical MLP `tr(Sigma_W)` (absolute, unnormalized within-class scatter
trace, SPEC3_ADDENDUM.md section 24), final/initial ratio, 3 seeds x 2
datasets (rerun with the new metric, `configs/stage19_classical_{mnist,
fmnist}_pca4.yaml`, identical architecture/hyperparameters to the original
Stage 19 Arm V2(a)):

| Dataset | Seed ratios (final/init) |
|---|---|
| MNIST | 29.607, 27.346, 36.749 |
| Fashion-MNIST | 31.370, 52.821, 163.381 |

All 6 seeds show tr(Sigma_W) INCREASING by 1-2 orders of magnitude, the
opposite of the preregistered prediction (< 0.5). This does not contradict
Stage 19's `classical_nc1` result (NC1 ratio ~0.515, feature-space
collapse CONFIRMED) -- NC1 is Sigma_W NORMALIZED by Sigma_B (between-class
scatter), and Sigma_B grows even faster than Sigma_W as ReLU-MLP weights
scale up during training. The classical config (`configs/stage19_classical_
{mnist,fmnist}_pca4.yaml`) sets `weight_decay: 0.0005` -- L2 regularization
is present, not absent, but at this strength it is evidently insufficient
to bound the absolute activation scale: no batchnorm/weight-norm constraint
exists in this architecture, and 5e-4 SGD-momentum weight decay does not
stop tr(Sigma_W) from growing 27-163x over training. The class means
separate proportionally faster than the within-class spread grows, so the
NC1 ratio still falls -- but the absolute scale of the representation is
weakly-constrained-yet-still-unbounded in practice, unlike a quantum
state's fixed-trace density matrix, whose invariance is structural (T5),
not merely a matter of regularization strength.

**Cross-domain framing.** Both the classical MLP and the VQC show their
respective NC1-type ratios falling primarily through BETWEEN-class
expansion, not within-class contraction: classically, Sigma_B grows faster
than Sigma_W (both increasing, ratio still falls); quantumly, z-space
Sigma_B^z grows while total_var (which upper-bounds the within-class part)
is flat-to-mildly-contracting (Stage 14/T2' -- the class-mean radius grows
~5x, STAGE12_FINDINGS.md P3/THEORY_NOTES T2', while beta absorbs CE's
margin pressure rather than the state norm itself). The quantum case
carries an additional structural guarantee the classical case lacks: T5's
exact Hilbert-Schmidt conservation law holds regardless of regularization,
architecture, or training length, whereas the classical MLP's approximate
"NC1 falls despite Sigma_W growing" pattern is an empirical consequence of
this specific architecture's optimization dynamics, not a theorem.

VQC (no-reupload, matched PCA-4 inputs) `total_var` max relative change,
reusing Stage 19 Arm V2(b)'s existing 10 checkpoints (no rerun):

| Dataset | Seed max-rel-change |
|---|---|
| MNIST | 4.48e-15, 7.03e-15, 5.69e-15, 8.34e-15, 8.01e-15 |
| Fashion-MNIST | 4.86e-15, 5.81e-15, 4.69e-15, 9.49e-15, 6.21e-15 |

CONFIRMED at machine precision, 10/10 seeds, exactly reproducing Stage 19
V1(i)/V2(b) (same runs, same threshold).

**Scoring: REFUTED** on the classical-side direction (ratio >> 1 in 6/6
seeds, not < 0.5); the VQC-side invariance leg is independently CONFIRMED.
Reported factually per CLAUDE.md ("report numbers, not collapse/no-collapse
claims") -- the headline contrast is not weakened by this: an unbounded
classical representation whose absolute scale visibly explodes, next to a
bounded quantum representation whose absolute scale is conserved to 1e-15,
is if anything a sharper illustration of T5's structural (not merely
empirical) invariance than a mild contraction would have been. The
directional prediction was simply wrong about which way "unconstrained"
cuts.

Figure: `figs/fig19_contraction_vs_rotation.png` -- left panel classical
tr(Sigma_W) vs epoch (log-y, both datasets), right panel VQC total_var vs
epoch (log-y, both datasets, no-reupload arm).

## Arm W3 -- presentation fixes

### Stage 19 Arm V2 comparative table, split by dataset (no pooling)

**MNIST**

| Approach | Seeds | Test acc | TPT | z-space NC1 ratio (final/init) | Full-space classification | Abs. contraction (final/init) |
|---|---|---|---|---|---|---|
| (a) Classical MLP | 3 | 0.936 | 2/3 | 0.458 | N/A (fixed input, not well-posed) | tr(Sigma_W) **31.2x increase** (Arm W2) |
| (b) VQC no-reupload, R0 | 5 | 0.937 | 0/5 | 0.0483 | invariant (4.5-8.3e-15) | total_var ~6.7e-15 (invariant) |
| (c) VQC reupload, R0 | 5 | 0.688 | 0/5 | 0.0046 | mildly contracting | total_var 2.64% contraction |
| (d) VQC R3 (trainable measurement) | 3 | 0.694 | 0/3 | 2.351 (**increase**) | mildly contracting | total_var 2.70% contraction |

**Fashion-MNIST**

| Approach | Seeds | Test acc | TPT | z-space NC1 ratio (final/init) | Full-space classification | Abs. contraction (final/init) |
|---|---|---|---|---|---|---|
| (a) Classical MLP | 3 | 0.933 | 3/3 | 0.572 | N/A (fixed input, not well-posed) | tr(Sigma_W) **82.5x increase** (Arm W2) |
| (b) VQC no-reupload, R0 | 5 | 0.920 | 0/5 | 0.0724 | invariant (4.7-9.5e-15) | total_var ~6.2e-15 (invariant) |
| (c) VQC reupload, R0 | 5 | 0.738 | 0/5 | 0.0046 | mildly contracting | total_var 2.97% contraction |
| (d) VQC R3 (trainable measurement) | 3 | 0.731 | 0/3 | 1.483 (**increase**) | mildly contracting | total_var 4.07% contraction |

Pooled (both datasets, 6/10/10/6 seeds respectively) numbers match
STAGE19_FINDINGS.md's original table exactly (test acc 0.935/0.928/0.713/
0.713 [pooled (c) and (d) coincide at 0.713 to 3 s.f.]; z-space ratios
0.515/0.060/0.0046/1.92) -- the split does not change any verdict, it
shows neither dataset drives the pooled numbers alone. The new "Abs.
contraction" column is the headline addition: it shows the classical
arm's raw representation scale is unconstrained and grows by 1-2 orders of
magnitude in BOTH datasets independently, while every VQC arm's full-space
scale is invariant-to-mildly-contracting in both datasets independently --
the dichotomy (unbounded classical vs. structurally bounded quantum) holds
per-dataset, not just in aggregate.

### Preregistration ledger recount (Stages 12-19, from scratch)

**Convention, stated before counting:** count each independently-verdicted
LEG as one item -- if a stage's own findings file assigns separate
CONFIRMED/REFUTED/PARTIAL verdicts to distinct sub-claims under one
preregistration header (e.g. Stage 19's V2(d): "full-space leg CONFIRMED"
+ "REFUTED on z-space direction" are two distinct verdict words attached to
two distinct sub-claims in the same status-table row), each leg counts
separately. A header with only one verdict word, however many sub-bullets
it discusses narratively, counts once. Applied uniformly to every stage,
not invented post-hoc to hit any target number.

**Per-stage tally (full itemization: `PREDICTIONS.md`, every
`STAGE*_FINDINGS.md`, and `RESULTS_V3.md` read in full):**

| Stage | Items | CONFIRMED | REFUTED | PARTIAL | UNSCOREABLE/factual | Running total |
|---|---|---|---|---|---|---|
| 12 | 5 | 3 | 1 | 1 | 0 | 5 |
| 13B | 3 | 1 | 0 | 2 | 0 | 8 |
| 14 | 6 | 3 | 3 | 0 | 0 | 14 |
| 14B | 5 | 4 | 1 | 0 | 0 | 19 |
| 15 | 6 | 4 | 0 | 1 | 1 | 25 |
| 16b | 3 | 2 | 1 | 0 | 0 | 28 |
| 17b (HR6, HN5) | 2 | 2 | 0 | 0 | 0 | 30 |
| 18 | 7 | 4 | 0 | 1 | 2 | 37 |
| 19 | 11 | 6 | 1 | 0 | 4 | **48** |

Arithmetic: 5+3+6+5+6+3+2+7+11 = 48. Grand total by bucket:
3+1+3+4+4+2+2+4+6 = **29 CONFIRMED**; 1+0+3+1+0+1+0+0+1 = **7 REFUTED**;
1+2+0+0+1+0+0+1+0 = **5 PARTIAL**; 0+0+0+0+1+0+0+2+4 = **7 UNSCOREABLE/
factual**. 29+7+5+7 = **48**.

(Stage 16, the base hardware arm distinct from 16b, contributes 0 items --
it has no corresponding `PREDICTIONS.md` preregistration section, unlike
16b; its "factual conclusion" restates SPEC3_ADDENDUM.md section 22's
pre-existing licensed claim rather than scoring a registered prediction,
consistent with `RESULTS_V3.md`'s own ledger, which has no Stage-16
non-b row either.)

**Reconciliation against the paper's claimed 35** (`paper_ieee/
NUMBER_AUDIT_IEEE.md` line 47: "35 preregistered items: 23 CONFIRMED, 6
REFUTED, 5 PARTIAL, 1 reported factually"): **discrepancy confirmed and
now fully accounted for, +13 items, none of it noise:**

| Bucket | Paper (35) | Recount (48) | Delta | Source |
|---|---|---|---|---|
| CONFIRMED | 23 | 29 | +6 | All from Stage 19 (V1(i), V2(a)-feature, V2(b), V2(c)-z, V2(c)-full, V2(d)-full) |
| REFUTED | 6 | 7 | +1 | Stage 19 V2(d)-z (NC1_z increased) |
| PARTIAL | 5 | 5 | 0 | No Stage 19 item lands here |
| UNSCOREABLE/factual | 1 | 7 | +6 | Stage 18 B6, B7 (+2, dropped from the paper's derivation despite being counted in RESULTS_V3.md's own ledger); Stage 19 V1(ii), V1(iii), V2(a)-full, V3 (+4) |

The paper's "35" traces exactly to Stages 12-17b's 30-item tally (19/6/4/1,
reproduced digit-for-digit by this recount) plus Stage 18's B1/B2/B3/B5
(CONFIRMED, +4) and B4 (PARTIAL, +1) -- but that derivation (a) silently
drops Stage 18's B6/B7 even though they are counted rows in `RESULTS_V3.md`
itself, and (b) never folds in Stage 19 at all. Separately, `RESULTS_V3.md`
itself has drifted since the paper's number was frozen: it now states its
own tally as 44 items (27/7/5/5), which is internally arithmetically
self-consistent (its 44 rows sum to exactly 27/7/5/5) but uses one-row-per-
header counting throughout rather than the leg-split convention used here,
and it also omits Arm V3 from its ledger table. Under a stricter reading
that excludes pure-deliverable items (B6, B7, V3 -- none of which are
directional predictions) from "preregistered predictions" entirely, the
total is **45 (29 CONFIRMED, 7 REFUTED, 5 PARTIAL, 4 UNSCOREABLE)**. Either
way, the paper's live "35" figure is stale and should be replaced with
48 (leg-split, primary) or 45 (deliverables excluded, stricter reading).

## Required factual notes

**(a) Stage 19's R3 z-space increase is confounded by underfitting, not a
clean measurement-learnability result.** Arm V2(d) (VQC, trainable
measurement, readout_family R3) reached TPT in 0/6 seeds (0/3 MNIST, 0/3
Fashion-MNIST) with mean test accuracy 0.713 (0.694 MNIST, 0.731
Fashion-MNIST) -- the same underfit test-accuracy floor as the reupload
arm (c). The z-space NC1 ratio INCREASE (mean 1.92 pooled; 2.35 MNIST,
1.48 Fashion-MNIST) is therefore reported as a genuinely REFUTED direction,
but the arm never reached the terminal training regime (TPT) any other
scored NC1_z-decay result in this codebase (Stage 12 P1, Stage 15 HN3) was
gated on. This result does not disentangle whether trainable measurement
layers actively work against z-space collapse, or whether the R3 ansatz
at L=16 with 2 measurement layers simply cannot fit these classes well
enough to reach the training regime where collapse is normally observed.
Both are live explanations; this codebase's own preregistered gating rule
(NC1_z decay is only claimed in TPT-reaching seeds) was never satisfied
here, so the result stands as a REFUTED direction on an underfit arm, not
as evidence about measurement learnability per se.

**(b) reupload=true's real-data full-space contraction (2.0-3.6%, mean
2.8% pooled; 2.64% MNIST, 2.97% Fashion-MNIST) is notably larger than the
0.73% found on synthetic blobs3 (HR1, Stage 14B).** The direction is the
same (small, consistent contraction, not invariance) and both are far
below any threshold that would call the encoding non-isometric, but the
magnitude is 3-4x larger on real image PCA-4 features than on the
synthetic blobs3 recipe. This is reported as an open quantitative gap, not
explained here -- real-data PCA-4 projections may sit further from the
regime where the reupload isometry approximation is tightest, but this is
not tested directly in this stage.

**(c) The arm with the strongest z-space collapse ties for the worst test
accuracy, with the architecture confound named explicitly.** Arm (c) (VQC
reupload, z-space NC1 ratio 0.0046 pooled, the strongest collapse of any
arm in Stage 19/20) and arm (d) (VQC R3, which does NOT show z-space
collapse at all -- ratio 1.92, an increase) are TIED at the worst pooled
test accuracy (0.713 each), both markedly below the no-reupload VQC arm
(b, 0.928) and the classical MLP (a, 0.935). The confound is architectural,
not a clean "more collapse -> worse accuracy" trend: (c) and (d) share the
same depressed accuracy despite opposite z-space behavior, so whatever
architectural choice they share (both are L=16, R0/R3 with reupload=true
respectively -- note (d)'s base config also sets reupload=true per
PREDICTIONS.md's Arm V2(d) definition) plausibly drives the accuracy gap
independent of the collapse mechanism. This stage does not isolate which
shared property (reupload's altered effective circuit depth/expressivity,
vs. something else) is responsible.
