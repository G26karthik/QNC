# STAGE16_FINDINGS.md — Hardware validation (optional arm)

Citations mandatory: Papyan, Han, Donoho (2020), PNAS (classical NC); Du,
Yang, Tao, Hsieh (2023), PRL 131, 140601 (quantum NC at the static optimum);
San Sebastian, Canizo, Orus (2026), arXiv:2602.08485 (empirical QNC study).

## Setup

Inference-only hardware arm per SPEC3_ADDENDUM.md section 22: the trained
n=4, L=8, R0 anchor (`transition_L` sweep, seed 0, reupload=true, final
checkpoint `epoch_0600.pt`, `results/transition_L_L8_s0_20260712-194651/`)
re-expressed in Qiskit and executed on IBM hardware, no training on
hardware, no tomography.

- **Circuit**: 4 qubits, 8 data-reuploading layers. Per layer: angle
  encoding (`RY(x_j)` per qubit, same 4-d PCA-scaled input re-uploaded every
  layer), then per-qubit `Rot = RZ(w[l,q,0])·RY(w[l,q,1])·RZ(w[l,q,2])`,
  then ring `CNOT(q, (q+1) mod 4)` for q=0..3. Readout: Z-basis measurement
  on qubits 0,1,2 (R0, C=3).
- **Entangler range**: verified against `qml.StronglyEntanglingLayers`
  source — because `VQC._circuit_body` (reupload path) invokes
  `StronglyEntanglingLayers` separately per layer with a length-1 weight
  slice, PennyLane's default range formula is evaluated fresh each call on
  `n_layers=1`, giving range=1 on every layer regardless of global layer
  index. Not assumed; read from source and pinned by
  `tests/test_stage16_translation_gate.py`.
- **Data**: `load_blobs3(60, blob_dist=3.0, blob_std=0.8, test_fraction=0.2,
  seed=0, ambient_dim=4)` + `pca_encode(n_qubits=4, seed=0)`, train+test
  concatenated (144+36=180 samples total — the full deterministic
  seed-0 dataset used by the anchor training run).
- **Backend**: `ibm_marrakesh` (156 qubits), selected as the operational
  backend with fewest pending jobs at submission time (0 pending; not
  `ibm_kingston`, which had 16 pending at the time — picked dynamically per
  the stated rule rather than the doc's anticipated default).
- **Transpilation**: preset pass manager, `optimization_level=3`,
  dynamical decoupling enabled.

## Budget used vs remaining

10 QPU-minutes / 28-day period, account total.

| Job | Circuits | Shots each | `qpu_charge_time_seconds` |
|---|---|---|---|
| Smoke (`d9cf73ineu4c739lt0r0`) | 5 | 1024 | 4 |
| Full batch (`d9cf942neu4c739lt300`) | 180 | 2048 | 104 |
| **Total used** | | | **108 s ≈ 1.8 min** |
| **Remaining this period** | | | **≈ 8.2 min** |

Smoke-job extrapolation (pre-submission estimate) was 3–5.5 min; actual full-batch
cost (104 s ≈ 1.7 min) came in well under that range.

## Task A: translation gate

`tests/test_stage16_translation_gate.py`: Qiskit statevector circuit vs the
existing PennyLane pipeline (`model.state(x)` -> `z_features`), all 180
samples. **Max abs difference: < 1e-6 (PASSED).** This was the hard gate;
no hardware job was submitted before it went green. Full suite: 166/166
passed before proceeding to Task B; 171/171 (adds Task A + hardware unit
tests) passing as of this writeup.

## Task C: hardware vs simulator comparison

180 samples, 2048 shots each, Z-basis measurement, binomial shot-noise
error propagation (`qnc.hardware.z_from_counts`).

| Metric | Hardware | Simulator (exact) |
|---|---|---|
| NC1_z | 0.6749 | 0.5472 |
| r_z (tr Σ_W^z / tr Σ_B^z) | 1.7981 | 1.5262 |
| d_box | 1.5549 | 1.4732 |
| mean_margin | 0.1619 | 0.2418 |
| min_margin | -0.4590 | -0.4511 |

Per-class z-space means (zbar_c, rows = class, columns = z[0],z[1],z[2]):

| | Hardware | Simulator |
|---|---|---|
| class 0 | [0.1550, -0.0781, -0.0457] | [0.2245, -0.1134, -0.0787] |
| class 1 | [-0.0874, 0.1512, -0.0815] | [-0.1394, 0.2229, -0.1171] |
| class 2 | [-0.0470, -0.1043, 0.1805] | [-0.0768, -0.1507, 0.2402] |

Figures: `results/stage16_hardware/d9cf942neu4c739lt300/figures/zspace_hw_vs_sim.png`
(z-space class-mean overlay), `.../nc1z_hw_vs_sim.png` (NC1_z bar comparison).

- Mean absolute z-difference (hardware − simulator): 0.0740; max: 0.2578.
- Mean binomial shot-noise std per z-component: 0.0217.
- Fraction of (sample, class) z-values agreeing with the simulator within
  ±1 shot-noise std: **17.6%**.

## Factual conclusion

The licensed claim (SPEC3_ADDENDUM.md section 22) is exactly: agreement or
disagreement with simulation within shot noise plus hardware error —
nothing stronger. Result: hardware z-values do **not** agree with the
exact simulator within shot noise alone (17.6% of components fall inside
±1σ shot-noise band; mean absolute deviation 0.074 is ~3.4x the mean
shot-noise std of 0.022). The direction and rough scale of the effect are
consistent with a uniform contraction of all z-components toward the
origin (all three diagonal zbar_c entries shrink hardware-vs-simulator by
a similar ~25-30% factor, off-diagonal entries shrink similarly), the
signature expected from decoherence/readout error rather than a
qualitative change in geometry: the sign pattern of zbar_c (positive
diagonal, negative off-diagonal) and the overall NC1_z order of magnitude
are preserved on hardware. NC1_z is numerically higher on hardware (0.675
vs 0.547 exact) — consistent with a noise floor that shrinks between-class
separation somewhat faster than within-class scatter, not with a collapse
or anti-collapse claim. No claim beyond these numbers is made.

## Commit

Raw hardware results (`results/stage16_hardware/`, including per-sample
counts and job logs) committed alongside code per CLAUDE.md invariant 3
(append-only, no overwriting).

=== STAGE 16 COMPLETE ===

---

# Stage 16b — contraction decomposition + untrained hardware control

Preregistered in `PREDICTIONS.md` (HW1-HW3) before any Stage 16b metric was
computed. Same anchor, same 180 samples, same backend family
(`ibm_marrakesh`), reusing Stage 16's existing counts for HW1/HW2 (zero new
hardware) plus one new job (HW3, untrained parameters).

## Budget used vs remaining

| Job | Circuits | Shots each | `qpu_charge_time_seconds` |
|---|---|---|---|
| Stage 16 smoke | 5 | 1024 | 4 |
| Stage 16 full (trained) | 180 | 2048 | 104 |
| Stage 16b HW3 (untrained, `d9cfo4ineu4c739ltj40`) | 180 | 2048 | 104 |
| **Total used this period** | | | **212 s ≈ 3.5 min** |
| **Remaining this period** | | | **≈ 6.5 min** |

Pre-submission estimate for HW3 was ~100-115s (same circuit shape/depth as
the Stage 16 full batch, only rotation angles differ); actual cost (104s)
matched that estimate exactly.

## HW1 — uniform-contraction fit (existing counts, no new hardware)

Least-squares `z_hw = lambda*z_sim + residual` (through origin, pooled over
all 540 (sample,component) pairs):

| Quantity | Value | Threshold |
|---|---|---|
| lambda | 0.7033 | [0.6, 0.85] — met |
| R^2 | 0.9305 | >= 0.9 — met |
| nc1_z predicted (lambda^2*Sigma_sim + Sigma_resid, pinv sandwich) | 758.71 | within 15% of 0.6749 — **missed by ~1123x** |

**Verdict: REFUTED.** The magnitude/fit-quality sub-predictions (lambda,
R^2) both held, but the NC1_z-prediction sub-prediction failed
catastrophically. Diagnosis: `Sigma_B^z` for any 3-class, 3-dim z-space is
exactly rank <= 2 by construction (class-mean deviations from the global
mean always sum to zero), so `Sigma_B^z_sim` has one EXACTLY null
eigendirection. The empirical residual's own `Sigma_B^resid` is generically
full-rank and does not share that null direction, so
`Sigma_B^z_hw_pred = lambda^2*Sigma_B^z_sim + Sigma_B^resid` picks up a
small but nonzero eigenvalue exactly where the simulator term contributes
nothing — `pinv`'s default tolerance does not treat that eigenvalue as
negligible (it is `O(residual_var)`, not floating-point noise), so the
trace-with-pinv formula divides by a small number and blows up. This is a
genuine failure of the specific two-term additive-covariance decomposition
formula preregistered for HW1, reported as such — not softened, not
patched post-hoc. It does not contradict HW2 below, which tests a
different (and simpler) claim.

## HW2 — angle invariance under contraction (existing counts, no new hardware)

`equiangle_dev_z`, hardware bootstrapped (1000 resamples, per-sample
Gaussian perturbation at each z-component's binomial shot-noise std) vs the
exact simulator:

| Quantity | Value |
|---|---|
| Hardware bootstrap mean +/- std | 0.0983 +/- 0.0082 |
| Hardware bootstrap 95% CI | [0.0831, 0.1152] |
| Simulator (exact) | 0.0969 |

**Verdict: CONFIRMED.** The simulator's exact value (0.0969) falls
comfortably inside the hardware bootstrap 95% CI — the angular geometry of
the z-space class-mean triangle is preserved under whatever is shrinking
its magnitude, consistent with HW1's confirmed lambda/R^2 (a real,
well-fit uniform scaling) even though HW1's specific NC1_z-prediction
formula is unstable. Figure:
`results/stage16_hardware/d9cf942neu4c739lt300/figures/zspace_hw_rescaled_vs_sim.png`
(hardware class means divided by the fitted lambda, overlaid on the
simulator's) shows this directly — the rescaled points visually coincide
with the simulator's across all 9 (class, component) entries.

## HW3 — untrained-parameter hardware control (one new hardware job)

Same 180 circuits, untrained seed-0 initial weights (reconstructed via
`torch.manual_seed(0)` immediately before `VQC(...)` construction, per this
repo's established epoch-0 convention — no `epoch_0000.pt` file was ever
saved, so this is a deterministic re-derivation, not a new artifact),
`ibm_marrakesh`, 2048 shots, job `d9cfo4ineu4c739ltj40`:

| Quantity | Value |
|---|---|
| NC1_z, untrained, hardware | 63.59 |
| NC1_z, untrained, simulator (exact) | 153.52 |
| NC1_z, trained, hardware (Stage 16) | 0.6749 |
| Ratio (untrained hardware / trained hardware) | 94.2x |

**Verdict: CONFIRMED** (ratio 94.2x >= the 10x threshold, by a wide margin).
The untrained circuit's hardware NC1_z is nearly two orders of magnitude
above the trained value — the Stage 16 hardware pipeline is measuring a
real trained-vs-untrained difference in the anchor circuit, not a
fixed hardware-noise floor independent of the parameters. (The untrained
simulator value, 153.52, is itself higher than the untrained hardware
value, 63.59 — both hugely elevated relative to either trained value,
consistent with near-zero between-class separation at initialization
making NC1_z's denominator small and the ratio noisy/large in either
setting; no claim beyond the ratio threshold is made.)

## Interpretation

The licensed claim remains exactly SPEC3_ADDENDUM.md section 22's:
agreement or disagreement within shot noise plus hardware error, nothing
stronger. HW1/HW2/HW3 do not upgrade that claim; they characterize its
structure. Net picture: hardware disagreement with the exact simulator (Stage
16) is well-described, in magnitude and angle, by a single scalar
contraction (lambda ~ 0.70, R^2 ~ 0.93, angles preserved within shot noise
per HW2) — but a naive additive-covariance model of how that contraction's
residual noise propagates into NC1_z itself is unstable and REFUTED (HW1),
because NC1_z's pinv-based formula is acutely sensitive to whichever
direction Sigma_B happens to be exactly degenerate in. HW3 confirms the
pipeline is not measuring a parameter-independent artifact.

## Commit

Raw hardware results for the new HW3 job
(`results/stage16_hardware/d9cfo4ineu4c739ltj40/`) and all Stage 16b
analysis summaries/figures committed alongside code.

=== STAGE 16B COMPLETE ===
