# SPEC3_ADDENDUM.md -- Phase 3 Mathematical Specification
# Read with SPEC.md and SPEC2_ADDENDUM.md, which remain in force.
# Same rule: implement these formulas exactly; never improvise.

---

## 17. Phase 3 hypothesis and positioning (agent must internalize)

Phase 2 established: VQCs interpolate (TPT) while ALL full-space collapse
witnesses (trace distance, Fisher ratio, purity, ETF geometry) remain at
untrained-null levels. Phase 3 tests the reconciling hypothesis:

**Measurement-subspace collapse (MSC):** cross-entropy depends on the state
only through the C-dimensional projection z = (<Z_0>,...,<Z_{C-1}>). Collapse
is therefore predicted to occur IN z-space (where the loss lives) and NOT in
the orthogonal "fiber" directions of Hilbert space (which receive no loss
gradient). Full-space witnesses sum variance over 4^n - 1 directions and are
dominated by the C-irrelevant fibers, explaining every Phase 1/2 null.

Secondary prediction (geometry): the z-space constraint set is the box
[-1,1]^C, not a sphere. CE with per-class Z readout drives class means toward
box vertices v_c (+1 in coordinate c, -1 elsewhere), NOT toward a simplex ETF.
Phase 3 runs this as an explicit model comparison.

Tertiary prediction (expressivity): as the measurement becomes more
expressive/learnable (approaching Du et al.'s setting of free measurement
operators), full-space collapse witnesses should strengthen. Measurement
learnability is the knob that interpolates between our nulls and Du's theorem.

Classical anchors to cite: Mixon et al. (UFM gradient flow suppresses
directions orthogonal to the collapse subspace BECAUSE features are free
parameters); Yang et al. NeurIPS 2022 (fixed ETF classifier still induces
collapse in unconstrained features); recent constrained-expressiveness work
showing collapse weakens to subspace/directional collapse with real inputs.
Our setting differs from all three: compact state manifold, fixed
rank-structured measurement, features not free.

PREREGISTRATION DISCIPLINE: Stage 12 writes PREDICTIONS.md BEFORE computing
anything, stating the expected direction of every metric below. Results are
then reported against those preregistered predictions.

## 18. z-space (measurement-subspace) collapse metrics

For sample i with state |psi_i>, define the logit-feature vector

```
z_i = ( <psi_i|Z_0|psi_i>, ..., <psi_i|Z_{C-1}|psi_i> )  in [-1,1]^C
```

recomputable from every stored checkpoint. Class means zbar_c, global zbar_G.

### 18.1 Within-class collapse in z-space (direct classical-NC1 analog)

```
Sigma_W^z = (1/N) sum_c sum_{i in c} (z_i - zbar_c)(z_i - zbar_c)^T
Sigma_B^z = (1/C) sum_c (zbar_c - zbar_G)(zbar_c - zbar_G)^T
NC1_z     = (1/C) tr( Sigma_W^z pinv(Sigma_B^z) )
```

Also report the scale-free scalar ratio r_z = tr(Sigma_W^z)/tr(Sigma_B^z).

### 18.2 Margin trajectory

Per-sample logit margin m_i = z_i[y_i] - max_{c != y_i} z_i[c].
Report mean_margin, min_margin, and beta * mean_margin (effective CE margin)
per logged epoch.

### 18.3 Geometry model comparison: box vertices vs simplex ETF

Box-vertex target: v_c in R^C with v_c[c] = +1, v_c[c'] = -1 for c' != c.

```
d_box = (1/C) sum_c || zbar_c - v_c ||_2
```

ETF target: centered means mtilde_c = zbar_c - zbar_G; compute pairwise
cosines and equiangle_dev_z against -1/(C-1) exactly as SPEC section 4.3 but with
Euclidean inner products in R^C. Normalized comparison score:

```
d_etf  = equiangle_dev_z  (dimensionless)
d_boxn = d_box / 2        (2 = max coordinate range, makes d_box dimensionless-ish)
```

Report both trajectories; the paper's claim rests on which target the means
approach, so ALSO log the raw zbar_c vectors per epoch for direct plotting.
Applies at C=3 only (C=2 remains degenerate for angle-based witnesses; d_box
is still valid at C=2 and should be reported).

### 18.4 Fiber variance decomposition (the decisive dichotomy witness)

Decompose within-class Hilbert-Schmidt scatter into the measured subspace and
its complement. Let {A_1,...,A_C} = {Z_0,...,Z_{C-1}} (pairwise commuting,
orthogonal under HS after removing trace overlap; verify <A_j, A_k>_HS =
delta_jk * 2^n by unit test and orthonormalize B_j = A_j / 2^{n/2}).

For each sample deviation D_i = rho_i - rho_bar_c:

```
D_i^meas  = sum_j <B_j, D_i>_HS B_j        (projection onto span{B_j})
D_i^fiber = D_i - D_i^meas
V_meas  = (1/N) sum_i ||D_i^meas||^2_HS
V_fiber = (1/N) sum_i ||D_i^fiber||^2_HS
fiber_fraction = V_fiber / (V_meas + V_fiber)
```

MSC prediction: V_meas decays post-TPT (tracks NC1_z); V_fiber stays at
null-baseline level; fiber_fraction -> 1. This single figure IS the paper's
headline if confirmed. Unit test: for D_i = B_0 (pure measured direction),
fiber_fraction = 0; for D_i orthogonal to all B_j, fiber_fraction = 1.

### 18.5 Required unit tests

| Test | Expected |
|---|---|
| NC1_z with z_i == zbar_c per class, distinct means | 0.0 |
| d_box with zbar_c exactly at vertices | 0.0 |
| equiangle_dev_z on hand-built C=3 ETF in R^3 | 0.0 |
| fiber decomposition on constructed cases (18.4) | 0 and 1 |
| <Z_j, Z_k>_HS orthogonality check at n=4 | delta_jk * 16 |

## 19. Measurement-expressivity sweep (Stage 13)

Four readout families, total trainable parameter count MATCHED across arms
(pad the circuit depth of leaner arms so M is equal; state the matched M in
configs):

- R0 (baseline, existing): fixed Z_c on first C qubits, learnable beta.
- R1: all 3 single-qubit Paulis on each of the first C qubits (3C features)
  followed by a trainable linear head W in R^{C x 3C} (+ bias); logits = W f.
- R2: all weight-1 Paulis on all n qubits plus all weight-2 ZZ pairs
  (3n + n(n-1)/2 features) with trainable linear head.
- R3 (Du-style learnable measurement): fixed Z_c readout preceded by a
  trainable "measurement unitary" block V(phi) of 2 SEL layers whose
  parameters are counted separately in logs (field "m_params_measurement").

For each arm: 5 seeds, blobs3, C=3, n=4, base depth L=8 (adjusted for
matching), 600 epochs. Log ALL SPEC2 witnesses plus ALL section 18 witnesses.

Prediction: full-space witnesses (qnc1_fisher, purity, overlap_offdiag)
strengthen monotonically R0 -> R3; z/feature-space NC1 collapses in all arms.
For R1/R2, the "measured subspace" of section 18.4 is span of the orthonormalized
measured Paulis of that arm (generalize the projector accordingly; the head
means logit-space NC is computed on f mapped through W).

## 20. Robustness arms for the no-full-collapse claim (Stage 14)

1. beta ceiling: rerun L=8 anchor with beta_max = 500, 5 seeds. If full-space
   witnesses move materially, the Phase 2 claim must be rescoped.
2. Long horizon: one run (seed 0) at L=8 for 5000 epochs, checkpoint every 50.
   Report full-space witnesses on log-time axis; test for slow drift
   (Spearman correlation of witness vs log-epoch post-TPT, report rho and p).
3. Label smoothing: CE with smoothing 0.1, 5 seeds at L=8 (classically
   accelerates NC2; tests whether ANY collapse pressure reaches full space).
4. Transition localization: L in {3,5,6}, 10 seeds each, TPT fraction curve
   (tightens the Phase 2 "somewhere between 9% and 38%" statement).
5. QFI locator fix: recompute QFI rank per SINGLE input (4 separate inputs,
   report each rank and their mean) at epoch 0 for L in {8,16,21,26,32};
   tests whether the single-state rank saturates near 255 where the batch-
   aggregated rank did not. Purely methodological; closes a Phase 2 loose end.

## 21. Scale replication (Stage 15)

n=6, C=3, blobs3 lifted to R^6 (same center distance/std recipe). Depth probe
L in {8, 16, 32, 64} (M = 18L), 3 seeds, to find the interpolating depth;
then 5 seeds at the smallest interpolating depth computing the FULL section 18 + 
SPEC2 witness set. Purpose: does the z-space-collapse / fiber-no-collapse
dichotomy survive at larger n (where fibers are relatively even larger --
prediction: dichotomy strengthens, fiber_fraction closer to 1).

## 22. Optional hardware arm (Stage 16, run only if user confirms QPU access)

Inference-only: load trained L=8 parameters, execute on an IBM backend,
estimate z_i per sample from shots (Z-basis measurements on readout qubits;
1024 shots/sample minimum), compute NC1_z, d_box, margins from hardware
estimates, compare to simulator values with shot-noise error bars
(binomial propagation). No training on hardware. No tomography. Log backend
name, date, calibration snapshot fields if available. The claim licensed is
strictly: "z-space collapse witnesses computed from hardware measurements
agree with simulation within shot noise" (or do not).

## 23. New JSONL fields

```
"nc1_z": float, "r_z": float, "mean_margin": float, "min_margin": float,
"d_box": float, "equiangle_dev_z": float, "zbar_c": [[float]],
"v_meas": float, "v_fiber": float, "fiber_fraction": float,
"readout_family": "R0|R1|R2|R3", "m_params_measurement": int,
"label_smoothing": float, "hardware_backend": str|null
```

## 24. Absolute within-class scatter trace (Stage 20, classical MLP arms)

`classical_nc1` (section 5, existing) reports the NORMALIZED ratio
tr(Sigma_W @ pinv(Sigma_B)) -- it cancels out the overall scale of the
feature space, so it cannot show whether the penultimate representation is
physically contracting or just changing shape. Stage 20 Arm W2 needs the
ABSOLUTE (unnormalized) within-class scatter trace as the direct classical
analogue of the VQC's `total_var` (total Hilbert-Schmidt within-class
variance, section 18): both are un-normalized traces of a within-class
second-moment object, one in feature space, one in density-matrix space,
so their final/initial ratios are directly comparable.

`within_class_scatter_trace(features, labels) = tr(Sigma_W)`, where
`Sigma_W = (1/N) sum_c sum_{i in c} (h_i - mu_c)(h_i - mu_c)^T` (identical
Sigma_W construction to `classical_nc1`, section 5 -- same accumulation,
different final reduction: trace alone, no pinv(Sigma_B) division). Always
>= 0 (a trace of a PSD matrix, a sum of per-class variances). Logged only
for classical MLP runs (model.type == "mlp"); VQC runs already log
`total_var` per checkpoint and have no feature-space analogue to compute
this from.

New JSONL field: `"within_class_scatter_trace": float | null` (null for
VQC runs).
