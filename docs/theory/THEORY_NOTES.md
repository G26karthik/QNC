# THEORY_NOTES.md — Formal statements (to be verified and owned by the user)

Every statement below is a claim to be checked, tightened, and proved by the
user; nothing here is a substitute for that proof. Each entry states the
claim, its exact scope of applicability, and the empirical numbers from
RESULTS_V3.md / the findings files that motivate or corroborate it. Mark:
**status: informal statement only, proof owned by the user.**

Citations mandatory throughout (CLAUDE.md): Papyan, Han, Donoho (2020),
PNAS. Du, Yang, Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo,
Orus (2026), arXiv:2602.08485.

---

## T1 — Cross-entropy factors through the C-dimensional readout

**Claim.** For the model family in SPEC.md section 2 / SPEC3_ADDENDUM.md
section 18 (Pauli-Z readout on the first C qubits, logits = beta·z), the
cross-entropy loss L(theta) depends on theta only through the map
theta -> z(theta) = (⟨Z_0⟩,...,⟨Z_{C-1}⟩) in R^C. Consequently, for any
perturbation direction H in the tangent space of density operators at
rho(x; theta) that is Hilbert–Schmidt-orthogonal to the measured span
B = span{B_0,...,B_{C-1}} (the orthonormalized Pauli-Z generators, SPEC3
section 18.4), the loss gradient in that direction vanishes:

```
dL/dtheta restricted to directions with <H, B_j>_HS = 0 for all j  =  0
```

**Scope.** Holds for any measurement/readout that is a fixed linear
functional of the state (any B, not just Pauli-Z); does not require
reupload=false. This is the general "loss sees only its own moment map"
statement — the mechanism argument (T5) is the specific reupload=false
strengthening of it.

**Corollary — no loss-selected pressure along fibers.** Directions
Hilbert-Schmidt-orthogonal to the measured span (the "fiber" directions of
SPEC3 §18.4) receive zero gradient at every step of training, for any
training trajectory, regardless of encoding: the loss selects no direction
of change along the fibers, so gradient descent applies no pressure there
at all, in either sign. Full-space collapse witnesses (Phase 1/2, e.g.
qnc1_fisher) sum variance over all 4^n − 1 traceless-Hermitian directions
and are therefore diluted by exactly this zero-gradient subspace whenever
C ≪ 4^n − 1 — this is the structural explanation for every Phase 1/2 null
(RESULTS_V2.md §2, §4) and motivates the z-space witnesses of SPEC3
section 18.

**Empirical corroboration.** Stage 12 P4 (fiber_fraction 0.9934, L=8
anchor); Stage 13B H1 (fiber directions sit at the isotropic rate,
complement/meas ≈ 1.06–1.16, i.e. no depletion outside the measured span);
Stage 15 HN2 (fiber_fraction strengthens toward 1 as n grows, tracking the
dimension count in T3).

**Status: formal statement only, proof owned by the user.**

---

## T5 — Collapse by rotation (centerpiece)

**Claim.** For angle encoding without data re-uploading
(`model.reupload: false`), each sample x is encoded once into a
theta-independent pre-image state rho_0(x) = |psi_0(x)⟩⟨psi_0(x)|, and the
entire trained circuit acts as a single global unitary applied after that
fixed encoding:

```
rho(x; theta) = U(theta) rho_0(x) U(theta)^dagger
```

Conjugation by a unitary is an isometry of the Hilbert–Schmidt inner
product: for any two operators A, B, Tr((U A U^dagger − U B U^dagger)
(U A U^dagger − U B U^dagger)^dagger) = Tr((A−B)(A−B)^dagger). Applying
this to every pair of sample states in the training set, U(theta) preserves:

- every pairwise Hilbert–Schmidt distance between rho(x_i; theta) and
  rho(x_j; theta), for every i, j, and every theta;
- consequently the entire within-class HS geometry of the constellation
  {rho(x_i; theta)} — in particular `total_var` = V_meas + V_fiber (SPEC3
  §18.4, summed over the full 4^n − 1 traceless-Hermitian directions via
  Parseval's identity) is **exactly invariant under training** in this
  setting.

**Corollary (the paper's central claim).** Since total_var cannot change,
any apparent collapse — NC1_z decay, the V_fiber dip below the
untrained-null level — is not a real contraction of the state
constellation. It is training driving U(theta) to ROTATE the fixed
constellation rho_0(x_1),...,rho_0(x_N) as a rigid body in Hilbert space,
redistributing a conserved variance budget between the measured subspace
(where the loss can act, per T1) and the fiber directions (where it
cannot). All collapse diagnostics under this architecture are, precisely,
rotation-only.

**Exact architectural scope (do not overgeneralize).**
1. Encoding applied exactly once, and the pre-image state rho_0(x) is
   parameter-independent (`reupload=false` in this codebase's terms).
2. The measurement (readout operators B_0,...,B_{C-1}) is fixed, not
   trained — a trainable measurement block (Stage 13's R3 family) would
   make even the *measured* projection itself rotate under training in a
   way not captured by this statement as written; T5 as stated covers R0
   only.
3. Data re-uploading (`_circuit_body`'s reupload branch, SPEC.md's
   `models.py`) breaks the argument outright: re-applying the encoding at
   every layer under the current trained weights makes the map from data
   to output state itself trained, not a fixed-then-rotated pre-image —
   real contraction becomes possible (Stage 14B HR1, HR4: reupload=true
   total_var declines, by a small amount at L=8 (0.73%) but significantly
   more with depth, rho=−0.451, p=0.0035).

**Machine-precision verification.** HR2 (STAGE14B_FINDINGS.md): max
relative change in total_var over 600 epochs, reupload=false, n=4, L=8, 5
seeds: 3.56e-15 to 5.84e-15 — nine orders of magnitude below the
preregistered 1e-6 threshold. HN1 (STAGE15_FINDINGS.md): identical
fingerprint at n=6, 5 seeds: 4.40e-15 to 6.54e-15. Both are numerical
confirmations of the isometry argument to floating-point precision, not
statistical tendencies.

**Corollary (purity).** Since unitary conjugation preserves Tr(rho^2) for
any operator rho (Tr((U rho U^dagger)^2) = Tr(U rho^2 U^dagger) = Tr(rho^2)
by cyclicity), and the same U(theta) acts on every class-mean state
rho_bar_c = (1/N_c) sum_{i in c} rho(x_i;theta) = U(theta) rho_bar_c^{(0)}
U(theta)^dagger (linearity of the average commutes with conjugation by a
fixed U), the per-class-mean purity Tr(rho_bar_c^2) is exactly invariant
under training in this setting, for every class c. Verified to machine
precision as HR6 (n=4, max relative change 3.26e-15 to 6.45e-15 across 5
seeds, CONFIRMED) and HN5 (n=6, 3.68e-15 to 6.17e-15 across 5 seeds,
CONFIRMED) — see STAGE14B_FINDINGS.md and STAGE15_FINDINGS.md.

**Corollary (Du-orthogonality witness) and structural non-reachability of
Du et al.'s optimum under this architecture.** By the identical argument,
the pairwise class-mean overlap Tr(rho_bar_c rho_bar_c') — the direct
density-matrix-space test of Du, Yang, Tao, Hsieh (2023)'s orthogonality
prediction (their global optimum requires class-mean measurement operators
to become mutually orthogonal, i.e. this overlap to approach 0) — is
likewise exactly invariant under training whenever reupload=false. Verified
to machine precision as the second HR6/HN5 quantity (n=4: 3.26e-15 to
6.12e-15; n=6: 3.94e-15 to 6.12e-15 relative change, both CONFIRMED).
**This yields a structural (not merely empirical) statement:** under fixed
encoding (no re-uploading) and fixed measurement (R0, untrained readout),
whatever pairwise class-mean overlap the untrained model starts with at
initialization is exactly the overlap it ends with after training — Du et
al.'s orthogonal-means optimum (overlap → 0) is training-unreachable in
this architecture unless the untrained initialization already happens to
sit at it. Reaching that optimum requires breaking the isometry, i.e.
either data re-uploading (parameter-dependent encoding, HR1/HR4) or a
trainable measurement block (Stage 13's R3 family, outside T5's scope per
its item 2 above) — collapse toward Du et al.'s specific target is
architecturally gated on giving up the exact-rotation guarantee that HR2/
HR3/HR6 rely on for the opposite (total-variance-conserving) claim.

**Caveat — entanglement entropy is NOT covered by this invariance (global
vs. local unitaries).** T5's isometry argument applies to any quantity
that is a basis-independent function of the full n-qubit state under a
GLOBAL unitary (Hilbert-Schmidt norms/distances, purity, pairwise
overlaps — all trace functionals of products of the full density
operators). Entanglement entropy at a FIXED bipartition (SPEC.md's
`entropy_mean` witness, one fixed cut) is not such a quantity: it is the
von Neumann entropy of a REDUCED state after tracing out a specific
subsystem, and is invariant only under LOCAL unitaries (a product
U_A ⊗ U_B factored along that same cut), not under a generic GLOBAL
unitary such as the SEL ansatz's entangling U(theta) (which mixes the two
factors via its CNOT layers). A global U(theta) can and generically does
change the entanglement entropy across any fixed cut even while leaving
every T5-covered quantity (total_var, purity, pairwise overlap) exactly
invariant. Do not extend "collapse by rotation, nothing changes" to
entanglement dynamics without separately verifying it — T5 says nothing
about `entropy_mean`'s trajectory, confirmed or otherwise.

**Status: formal statement only (a short, clean lemma given standard
unitary-conjugation HS-isometry facts, plus the two corollaries above
following from the same cyclicity-of-trace argument), proof owned by the
user.**

---

## T2′ — Class means adopt simplex-ETF directions without box saturation; learnable beta absorbs the scale

**[CORRECTED, Stage 17b — see STAGE12_FINDINGS.md's P3 correction note for
the original framing this replaces.]**

**Claim.** With a learnable inverse-temperature (scale) beta acting on the
logits (logits = beta·z, z in [-1,1]^C, per-class Z readout), the trained
class means' DIRECTIONS (centered mtilde_c = zbar_c − zbar_G) converge to
the simplex-ETF angular target (pairwise cosine −1/(C−1)), while their
MAGNITUDE (radius ‖mtilde_c‖) stays far below the box-vertex saturation
radius. This is a sharper claim than "ETF beats box" (T2′'s original
framing): at C=3 the two geometric models are NOT angularly
distinguishable in the first place. The centered box vertices v_c (+1 in
coordinate c, −1 elsewhere) have pairwise cosine EXACTLY −1/(C−1) = −1/2
themselves (verified algebraically, `test_box_vertices_c3_cosine_equals_etf_target`,
`tests/test_metrics.py`) — any 3 non-collinear equidistant points in R^3
form a valid 2-simplex, and the box vertices happen to be one instance of
it at C=3. So a small `equiangle_dev_z` (d_etf) does not by itself refute
the box-vertex model; it only shows directions are simplex-aligned, which
box vertices also are. The discriminating test is RADIUS: box vertices
have a fixed nonzero saturation magnitude (‖v_c − v̄‖ = sqrt(8/3) ≈ 1.6330
for C=3), whereas an ETF target under a learnable, freely-rescaling beta
has no fixed radius at all — CE's pressure toward better-separated logits
is satisfied by growing beta, not by growing ‖z‖ itself, so the radius is
free to converge wherever gradient descent happens to leave it, generally
far short of the box's fixed corner.

**Empirical anchors.**
1. Angular convergence: Stage 12 P3 (STAGE12_FINDINGS.md), C=3, L=8 anchor,
   final epoch: d_etf (equiangle_dev_z) = 0.0722 ± 0.0245 — directions are
   close to the shared ETF/box-vertex angular target.
2. Radius trajectory (recomputed this pass, existing L=8 anchor checkpoints,
   5 seeds, no new training, via `reanalyze_l8_anchor_zspace`): mean class-mean
   radius ‖mtilde_c‖ grows from 0.0649 ± 0.0164 (epoch 0) to 0.3287 ± 0.0257
   (final epoch) — a ~5x increase, but the final value is only ~20.1% of
   the box-vertex saturation radius (sqrt(8/3) ≈ 1.6330). Directions align
   with the shared angular target while magnitude stays far from box
   saturation — exactly the "ETF directions without box saturation" claim.
3. Learnable beta absorbs the scale: Stage 14 arm 1 (STAGE14_FINDINGS.md) —
   raising the beta ceiling from ~30 (max observed trained value) to 500
   changes nothing (bit-identical trajectories, p=1) because the ceiling
   was never binding; separately, frozen beta (arm 6, beta=5.0 fixed, not
   learnable) delays TPT onset by ~4.1x (mean E0 145.2 vs 35.2 epochs) —
   consistent with beta's freedom to grow being what lets CE reach its
   loss-sensitive (small-radius-but-well-separated) regime quickly, rather
   than being a mechanism that forces additional within-class collapse
   (that specific causal claim was separately REFUTED, arm 6, wrong
   direction on trace_w). The radius staying small (point 2) while beta
   grows large is the mechanism: CE's margin pressure is satisfied by
   beta·radius growing, and beta is the cheaper (unconstrained, scalar)
   parameter to grow, so the radius itself is never pushed toward the
   box's fixed corner.

**Status: formal statement only (a box-constrained CE optimum-selection
argument distinguishing angle from radius — a clean lemma given standard
max-margin/ETF facts for classical CE, adapted to the box-constrained
z in [-1,1]^C setting with a free logit scale), proof owned by the user.**

---

## T3 — Dimension counting: fibers vs measured subspace, n-scaling

**Claim.** The measured subspace B = span{B_0,...,B_{C-1}} (SPEC3 §18.4,
orthonormalized Pauli-Z generators) has dimension exactly C (here C=3);
the ambient space of traceless Hermitian operators on an n-qubit Hilbert
space has dimension 4^n − 1. The fiber (HS-complement) subspace therefore
has dimension 4^n − 1 − C. Under an isotropic-null assumption (variance
spread uniformly over all 4^n − 1 directions, e.g. at random
initialization), the expected fiber fraction is:

```
fiber_fraction_null(n) = 1 - C / (4^n - 1)
```

which is strictly increasing in n (for fixed C): at n=4,
fiber_fraction_null = 1 − 3/255 ≈ 0.9882; at n=6, 1 − 3/4095 ≈ 0.99927.
**Predicted ordering:** fiber_fraction at n=6 > fiber_fraction at n=4, both
at the untrained null and (if the measured-subspace collapse mechanism
holds, T1/T5) at the trained/post-TPT state, since collapse only concentrates
variance loss within the already-small C-dimensional measured subspace,
never in the (relatively larger, as n grows) fiber subspace.

**Empirical corroboration.** Stage 12 P4 (n=4, L=8 anchor, final):
fiber_fraction = 0.9934. Stage 15 HN2 (n=6, L=8 campaign, final,
5-seed means): 0.99963 (reupload=true), 0.99917 (reupload=false) — both
exceed the n=4 reference (0.9934), and n=6 > n=4 as predicted. Stage 13B's
k-generalized fiber_fraction (R1 k=9, R2 k=18 at n=4) also tracks
1 − k/(4^n−1) closely (STAGE13B_FINDINGS.md Task A), the same formula with
C generalized to k.

**Status: formal statement only (a direct dimension count, not requiring
further proof beyond stating the isotropic-null assumption precisely and
verifying it empirically as done above), proof owned by the user.**

---

## T6 — NC1_z / pinv instability when feature dimension equals C (methodological)

**Claim.** NC1_z = (1/C)·tr(Sigma_W^z · pinv(Sigma_B^z)) is numerically
unstable whenever Sigma_B^z is exactly or near-exactly rank-deficient
relative to its own ambient dimension C. For any C-class problem with
per-class Z readout, Sigma_B^z ∈ R^{C×C} is *exactly* rank ≤ C−1 by
construction (class-mean deviations from the global mean zbar_G always sum
to zero: sum_c (zbar_c − zbar_G) = 0). Consequently pinv(Sigma_B^z) is
computed against a matrix with at least one exactly-zero eigenvalue; if any
perturbation to Sigma_B^z (empirical noise, hardware residual) introduces a
small-but-nonzero eigenvalue in that exact null direction, `pinv`'s default
tolerance does not treat it as negligible (it is O(perturbation variance),
not floating-point noise), and the trace-with-pinv formula divides by a
small number, blowing up.

**Recommendation.** Report the scale-free scalar ratio
r_z = tr(Sigma_W^z) / tr(Sigma_B^z) alongside NC1_z in any setting where
Sigma_B^z may be near-degenerate: small C (especially C ≤ 4, where the
C−1 ≤ 3-dimensional null structure is a large fraction of the ambient
space), or any measurement with empirical/hardware noise perturbing an
otherwise-exact rank deficiency. r_z has no matrix inversion and is
well-defined and well-behaved wherever NC1_z's pinv sandwich is not.

**Empirical corroboration.** Stage 16b HW1 (STAGE16_FINDINGS.md): the
naive additive-covariance NC1_z-prediction formula
(lambda^2·Sigma_B^sim + Sigma_B^resid, pinv sandwich) predicts NC1_z=758.71
against an actual hardware value of 0.6749 — off by ~1123x — precisely
because Sigma_B^resid (the empirical hardware residual's between-class
covariance) is generically full-rank and does not share Sigma_B^sim's exact
null direction. This REFUTES the specific two-term additive-covariance
NC1_z-prediction sub-claim of HW1 while leaving its magnitude/fit-quality
sub-claims (lambda=0.7033, R^2=0.9305) and the angle-invariance result
(HW2, CONFIRMED) unaffected — those do not route through pinv.

**Status: formal statement only (a rank/perturbation-sensitivity argument
for the pinv sandwich — straightforward given standard perturbation theory
for pseudoinverses near rank changes), proof owned by the user.**

---

## Summary table

| ID | One-line claim | Formal status | Key corroborating number |
|---|---|---|---|
| T1 | no loss-selected pressure along fibers (CE gradient vanishes off the measured span) | to be proven by user | fiber_fraction 0.9934 (n=4), 0.9996 (n=6) |
| T5 | reupload=false ⇒ unitary conjugation ⇒ HS-isometry ⇒ rotation-only collapse; purity/overlap invariant too; entropy NOT covered | to be proven by user | total_var ~1e-15 (HR2)/~5e-15 (HN1); purity+overlap ~1e-15 (HR6/HN5) |
| T2′ | class means adopt ETF directions without box saturation; beta absorbs scale | to be proven by user | d_etf 0.0722; radius 0.329 vs box-saturation 1.633 (~20%); frozen-beta E0 ~4.1x delay |
| T3 | fiber dimension count, n-scaling | to be proven by user | fiber_fraction n=6 (0.9996) > n=4 (0.9934) |
| T6 | NC1_z/pinv unstable at rank C−1; use r_z | to be proven by user | HW1 NC1_z-prediction off by ~1123x |
