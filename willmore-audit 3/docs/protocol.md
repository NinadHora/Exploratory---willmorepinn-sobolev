# Willmore Audit Protocol

30 September 2026

A neural Willmore surface is trusted only when identities that every genuine surface must satisfy hold on a grid the training never saw. This protocol fixes, for each certificate, what is computed, the pass rule with its threshold, what a failure means, and the adversarial test attached to it. It is the written contract behind the harness in willmore-audit and behind the presentation slides.

## 1. What trusted means here

The rule: a surface produced by the flow is accepted only when every implemented certificate passes on the frozen evaluation grid, and the certified energy agrees with the reported energy within Monte Carlo noise. Low training loss on its own is not evidence.

Two numbers are kept apart throughout, because the whole project lives in the gap between them.

| Name | Definition | Who computes it | Weakness |
| --- | --- | --- | --- |
| Reported energy, W\_rep | domain\_area × mean of H²√g over 5000 uniform random parameter points, with the Huber cap on H² at 50 | the training loop of Hirst, Sá Earp and Silva \[15\] | random samples miss small features; the Huber cap hides large curvature; the value moves with the seed |
| Certified energy, W\_cert | ∫ H² dA by spectral quadrature (midpoint rule in periodic directions, Gauss-Legendre in the polar direction) in float64, on a grid refined until Gauss-Bonnet closes | the harness (certificates.py) | costs more points; needs a grid that resolves the surface, which the resolution protocol in section 10 guarantees |

Each certificate is a theorem turned into a computation. The theorem says what a true Willmore surface must satisfy. The computation says whether this particular network output does. The gap between W\_rep and W\_cert is not a certificate; it measures how far the training signal can be pushed from the truth, and it is the quantity the attacks maximise.

The word certificate is used strictly. A certificate is an identity or inequality with an exact reference value, computed on data the training never touched. A diagnostic (residual size, seed spread) has no exact reference value and is reported as a number, not as pass or fail.

## 2. Objects

The harness audits any map φ from the parameter domain to R³ that torch can differentiate. Three kinds of φ enter: a network checkpoint, an exact formula (the unit tests), and a perturbed surface (the attacks). The harness does not know which one it is looking at.

**Domains.** Genus 0 uses u in \[0, 2π\] (azimuth, periodic) and v in \[0, π\] (polar). Genus 1 uses u and v both in \[0, 2π\], both periodic. These are the conventions of the published code, so a checkpoint plugs in without reparametrisation. Genus 2 is two glued torus charts with a disc removed from each; the harness does not support it yet (section 14).

**Curvatures.** First and second fundamental forms come from forward-mode automatic differentiation of φ (jacfwd, nested twice), exactly as in the published loss. H = (EN − 2FM + GL) / 2(EG − F²), K = (LN − M²)/(EG − F²), dA = √(EG − F²). Nothing is clamped in the certified path; a NaN is a finding (the metric degenerated), not a bug to hide.

**Frozen evaluation grid.** A tensor grid of n×n nodes: midpoint rule in each periodic direction, Gauss-Legendre in the polar direction of the sphere. Midpoint nodes never coincide with training samples, and Gauss-Legendre nodes never touch the poles. The default is 96×96 (9216 nodes); the resolution protocol (section 10) raises it when needed.

**Training estimator replica.** willmore\_mc reproduces the published loss: 5000 uniform parameter points, domain\_area × mean(H²√g), with √g floored at 10⁻⁶ and the Huber form on H² above 50 (exact below, linear in |H| above). It returns both the uncapped value and the Huber value, so the effect of the cap is measurable.

**Precision.** Everything runs in float64. Checkpoints are converted with model.double(); the spectral feature buffers convert with the model.

## 3. Unit tests: the exact surfaces

Every certificate must pass on the exact sphere and on the Clifford torus to machine precision before it is applied to a network. The numbers below were reproduced independently in numpy with fourth-order finite differences (validate\_numpy.py); the same values are the expected values in tests/test\_certificates.py. If the torch harness disagrees with them, the defect is in the autodiff path or in φ, not in the quadrature.

| Surface | Domain | W expected | W obtained | ∫K dA expected | ∫K dA obtained | Willmore residual (relative) |
| --- | --- | --- | --- | --- | --- | --- |
| Sphere, any radius | genus 0 | 4π = 12.5663706144 | 12.5663706143 | 4π | 12.5663706143 | 7.6e-7 (critical) |
| Ellipsoid (1, 1, 2) | genus 0 | above 4π | 15.4516066443 = 1.2296 · 4π | 4π | 12.5663706143 | not critical |
| Clifford torus, R = √2, r = 1 | genus 1 | 2π² = 19.7392088022 | 19.7392088023 | 0 | 4.6e-11 | 1.6e-6 (critical) |
| Torus R = 3, r = 1 | genus 1 | π²R² / (r√(R² − r²)) = 31.4048888984 | 31.4048888987 | 0 | 2.7e-10 | 0.81 (not critical) |

The fourth row matters as much as the third. A torus of the wrong aspect ratio is a valid closed surface, so Gauss-Bonnet closes, but it is not a Willmore surface, so the residual is order one. That separates certificate C1 (is it a torus) from certificate C4 (is it a Willmore torus).

All four surfaces also pass C2: after an inversion centred at (0, 0, 6) with radius 3, W and ∫K dA are preserved to 1e-8 on a 128×128 grid.

## 4. C1 Gauss-Bonnet: is the topology what it claims to be?

**Statement.** For a closed surface of genus g, ∫ K dA = 2πχ = 4π(1 − g). This holds for every smooth closed surface, Willmore or not. It is the cheapest certificate: K uses the same second derivatives as H.

**Computation.** total\_curvature integrates K√g on the frozen grid. The harness reports the estimate χ̂ = ∫K dA / 2π and the residual |∫K dA − 2πχ|.

**Pass rule.** Residual below 10⁻³ · 2π (about 6·10⁻³) on a grid that the resolution protocol has declared converged. The threshold is loose on purpose: the exact surfaces give 10⁻⁹⁰, a tanh network in float64 should give 10⁻⁶ or better, and anything near 10⁻³ already signals a defect. Genus 0 must give χ̂ = 2, genus 1 must give χ̂ = 0.

**What a failure means.** Read χ̂ together with its behaviour under refinement.

| χ̂ behaviour | Reading | Next step |
| --- | --- | --- |
| Integer, correct, stable under refinement | topology certified | proceed to C2 |
| Non-integer, changes as the grid is refined | grid does not resolve the surface (a thin feature, a spike) | refine until stable; report the grid size |
| Non-integer, stable under refinement | the map is not a smooth closed surface: a pinched neck, a degenerate metric, a tear at a seam | inspect dA minima; for genus 2, inspect the gluing collar |
| Integer but wrong | the network changed topology (a torus that pinched to a sphere gives χ̂ = 2) | this is the failure the adaptive safeguards of the training were meant to catch |

**Attack attached.** A thin grafted bubble (section 8) does not change χ, so a correct C1 on a coarse grid is exactly where the bubble hides. Observed on the Clifford torus with a bubble of height 0.2 and parameter width 0.15 at the inner equator: ∫K dA = −1.69 at 128², −0.44 at 256², −0.10 at 384². The certificate fails first and W converges later, which is why C1 is the resolution criterion in section 10.

**Status.** Implemented, validated on the exact surfaces, to be run on the genus 0 and genus 1 checkpoints.

## 5. C2 Möbius invariance: a thermometer with an exact zero

**Statement.** W(Ψ ∘ φ) = W(φ) for every Möbius transformation Ψ of R³ ∪ {∞} whose centre is off the surface (Blaschke \[2\], White \[3\]). Pointwise, (H² − K) dA is conformally invariant, and ∫K dA is topological, so both halves are preserved separately.

**Computation.** mobius\_inversion composes φ with Ψ(x) = c + r²(x − c)/|x − c|² and stays differentiable, so the same autodiff path computes W on the inverted surface. The centre defaults to a point above the bounding box at 1.5 box diagonals; the radius defaults to the mean distance from the centre to the surface, so the inverted surface has a size comparable to the original. A second run with the centre at the centroid (inside the surface) is recommended: it produces a different distortion and catches a coincidence.

**Pass rule.** |W(Ψ∘φ) − W(φ)| / W(φ) below 10⁻³, and ∫K dA on the inverted surface within the C1 tolerance. On the exact surfaces the relative difference is 10⁻⁹ to 10⁻⁸.

**Why this is a thermometer.** The identity is exact, so any difference comes from discretisation and from the estimator, never from the geometry. Inverting stretches some regions and compresses others, so the two grids sample the same surface at different densities. A difference that shrinks under refinement is quadrature error; a difference that persists is a real defect in φ (a region where the metric is nearly degenerate and the inversion amplifies it).

**What a failure means.** Relative difference above 10⁻³ with C1 passing: the surface is a closed surface, but W is not yet resolved; refine. Relative difference above 10⁻³ with C1 failing: go back to C1. A difference that is large and grows under inversion centred inside the surface: the surface passes near the centre, or self-intersects there; move the centre and rerun; a self-intersection is then a case for C3.

**Attack attached.** The same identity can be applied to the training estimator: W\_rep(Ψ∘φ) versus W\_rep(φ) over seeds. The spread of that difference is a direct measurement of the Monte Carlo error with an exact reference value of zero, which is what section 11 uses to size the ties.

**Status.** Implemented and validated.

## 6. C3 Li-Yau and embeddedness: the decision table

**Statement.** If φ covers some point of R³ k times, then W(φ) ≥ 4πk (Li and Yau \[4\]). So a certified W below 8π = 25.13 forces the surface to be embedded. The genus 1 target 2π² = 19.74 is below 8π; the genus 2 value of 29.70 is above it, so for genus 2 the theorem gives no embeddedness at all.

**Computation.** li\_yau reports the multiplicity bound ⌊W\_cert / 4π⌋ and the flag W\_cert < 8π. The second half, a direct self-intersection test, builds a triangle mesh from a regular (u, v) grid of φ and runs triangle-triangle intersection on it (planned for week 7; not in the current harness).

**Pass rule.** There is no single pass. The two tests are read together.

| Mesh self-intersects? | W\_cert < 8π? | Reading |
| --- | --- | --- |
| no | yes | embedded and certified; the theorem and the mesh agree |
| no | no | embedded, but the theorem cannot confirm it; only the mesh test speaks (this is the genus 2 case) |
| yes | no | immersed with a genuine self-intersection; consistent with the theorem; not a valid competitor for the embedded minimum |
| yes | yes | contradiction: the theorem forbids this, so the certified energy is wrong, or the mesh test found a numerical near-touch; refine both and rerun |

The last row is the important one. A self-intersecting output with reported energy below 8π is a proof that the estimator is wrong, not that the theorem fails. This is the only certificate that can convict the estimator outright rather than measure it.

**Attack attached.** Near-self-intersecting tori below the regularity thresholds of the published config: area element floor 0.01, E and G between 0.001 and 5, mean area floor 0.2. A torus whose tube almost touches itself satisfies all of these and has W\_rep well below 8π. The question is whether the mesh test catches the touch before the thresholds do.

**Status.** The Li-Yau half is implemented. The mesh half is week 7.

## 7. C4 The Willmore equation: is it a critical point at all?

**Statement.** Critical points of W satisfy ΔH + 2H(H² − K) = 0, with Δ the Laplace-Beltrami operator of the induced metric \[1\]. Low energy says the surface is near the bottom of the landscape; a small residual says the gradient is zero there. Both are needed: a run can stall at low energy on a slope (residual large), and a run can sit exactly on a critical point that is not the minimum (residual small, energy above the known value, the case that C7 then decides).

**Computation.** In coordinates, ΔH = (1/√g) ∂ᵢ(√g gⁱʲ ∂ⱼH). The harness defines H as a function of (u, v) by nested forward-mode autodiff (second derivatives of φ), differentiates it once more for ∂H (third derivatives), builds the flux √g gⁱʲ ∂ⱼH, and takes its divergence (fourth derivatives). Everything is autodiff; no finite differences enter.

**Scale.** The residual is reported in L² with weight dA, and also relative to a scale. The scale is the larger of the L² norm of |ΔH| + |2H(H² − K)| and H\_rms³. The second term exists because on the round sphere both terms of the equation vanish identically, so a ratio of the two would be 0/0; H\_rms³ gives the natural size of each term for a surface of that curvature.

**Thresholds.** Three bands rather than one line, because fourth derivatives of a tanh network are noisy and the band matters more than the digit.

| Relative residual | Band | Reading |
| --- | --- | --- |
| below 0.05 | critical | consistent with a Willmore surface at this grid; exact surfaces give 10⁻⁶ |
| 0.05 to 0.5 | near-critical | training stopped on a shallow slope, or the fourth derivatives are noisy; check with the fallback below |
| above 0.5 | not critical | the surface is not a Willmore surface, whatever its energy; the torus R = 3 gives 0.81 |

**Fallback when fourth derivatives are noisy.** The normal component of the L² gradient of W is exactly the left-hand side of the equation, so the residual can also be estimated as the first variation of W under a normal perturbation: perturb φ by ε ξ n with ξ a smooth test function on the grid, and read the slope of W\_cert in ε. This uses only second derivatives and a difference quotient in ε. It is coarser, but it agrees with the direct residual on the exact surfaces and is the number to report when the direct one blows up.

**Attack attached.** None needed. C4 is where the flow outputs get tested directly: a checkpoint with W near 2π² and a residual in the third band is a surface that the training accepted and the equation rejects.

**Status.** Implemented (direct residual). Validated: sphere 7.6·10⁻⁷, Clifford 1.6·10⁻⁶, torus R = 3 0.81. Fallback to be added if the network residual is unstable.

## 8. C5 Energy quantisation and the grafted bubble

**Statement.** Along sequences of surfaces with bounded energy, curvature can concentrate at points, and each concentration point carries at least 4π of energy (Simon \[7\], Rivière \[8\]). A small sphere grafted onto a surface costs about 4π of true energy and occupies almost no parameter area. This is not a certificate but the precise description of the failure mode of a uniform Monte Carlo estimator, and it is the blueprint for the attack.

**Construction.** graft\_bump builds φ\_ε(u, v) = φ(u, v) + ε · exp(−d²/ρ²) · n(u, v), with d the periodic distance to the centre (u₀, v₀), ε the height along the normal and ρ the width in parameter space. The physical width is ρ times the metric factor: on the torus, ρ·(R + r) = 2.42ρ along u at the outer equator, ρ·(R − r) = 0.41ρ at the inner equator, and ρ·r = ρ along v. The same parameter width is six times sharper on the inside of the torus than on the outside, which is the knob that controls how much the estimator misses.

**Two ways the estimator misses, and the regime of each.**

1. Sampling blindness. With 5000 uniform points over a parameter area of 4π², the expected number of samples inside a disc of radius ρ is 5000 · πρ² / 4π² ≈ 400 ρ². That is 100 points at ρ = 0.5, 9 points at ρ = 0.15, and one point at ρ = 0.05. Below ρ ≈ 0.1 the reported energy depends on whether a handful of samples happened to land on the bubble, so the seed spread grows before the mean drifts.
2. The Huber cap. The training loss replaces H² by a linear function above H² = 50, that is above |H| ≈ 7.07. A bubble of height ε and physical half-width w has curvature of order ε/w², so the cap bites once ε/w² exceeds about 7. At the inner equator with ρ = 0.15 (w ≈ 0.06 along u) even ε = 0.03 is enough; at the outer equator the same ρ needs ε of order 1.

**Measured on the Clifford torus (numpy validation, 192² or finer grids, two seeds).**

| ε | ρ | Where | W\_cert | W\_rep seed 0 / seed 1 | Huber value | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.5 | outer equator | 19.833 | 19.818 / 20.112 | same as W\_rep | wide bubble, well sampled; seed spread 0.3 already |
| 0.3 | 0.5 | outer equator | 20.351 | 20.473 / 20.758 | same | still resolved; seed spread 0.3 |
| 0.2 | 0.15 | outer equator | 22.816 | 23.897 / 22.792 | same | 9 samples on the bubble; seed spread 1.1; cap not reached |
| 0.2 | 0.15 | inner equator | above 46 (grid not converged at 384²; ∫K dA still −0.10) | 38.76 | 25.93 | cap active: the loss under-reports by more than 20 |

The last row is the regime the attack targets. The reported value the optimiser sees (25.9) is a third below the uncapped value it does not see (38.8), which in turn is far below the true energy. The grid did not converge there, so the certified number is a lower bound and the row is reported as such.

**Bubble radius in the sense of the theorem.** A grafted sphere of radius ε in the strict sense (a spherical cap sewn onto a disc) costs 4π minus the cap area correction. The Gaussian bump is a smooth stand-in whose energy grows continuously with ε/ρ²; the strict version is the annealing rung of the attack, where the search can push the profile toward the quantum.

**Status.** Implemented (graft\_bump, attack\_bump.py). The table above is from the numpy mirror; the torch run on the checkpoints fills Figure 1.

## 9. C6, C7, C8: planned certificates, with their criteria fixed now

These three are not in the current harness. Their pass rules are written here so that, when they run, the numbers are judged by a rule set before seeing them.

**C6 The true flow (Kuwert and Schätzle \[9\]).** The L² gradient flow of W decreases energy monotonically. Training is a gradient flow in the pullback metric of the network, not in L², so monotonicity is not guaranteed, and the published run logs only the reported energy. Criterion: compute W\_cert on every saved checkpoint of a run and plot it against the epoch beside W\_rep. Pass if W\_cert is non-increasing up to the seed spread measured in section 11. A run where W\_rep decreases while W\_cert rises is the optimiser descending on the estimator rather than on the energy; that is the finding, not a numerical accident. Data needed: checkpoints every 50 epochs, which the published loop already saves.

**C7 Second variation (Weiner \[11\]).** The Clifford torus is a stable critical point. For a run that stalls above 2π² with a residual in the critical band, the Hessian of W\_cert with respect to the network parameters, restricted to normal variations, decides between a saddle (some negative eigenvalue beyond the gauge directions) and a genuine non-minimal Willmore torus (none). Computation: Hessian-vector products by autodiff, Lanczos for the lowest 20 eigenvalues, with the Möbius directions (10-dimensional group, of which the ones that move the surface) and the reparametrisation directions projected out. Criterion: report the index only when the gauge projection is clean, that is, when the projected-out directions have eigenvalues within 10⁻⁶ of zero and the next one is separated by at least a factor 100. Otherwise report the spectrum without an index claim.

**C8 Discrete Willmore energy (Bobenko \[13\]).** Bobenko's energy on a triangle mesh is Möbius invariant by construction and needs no derivatives, so it is a second estimator with different failure modes. Criterion: build the mesh from the same regular grid used for the self-intersection test, compute W\_B, and compare with W\_cert. Agreement within 2% on the exact sphere and Clifford torus at 96² (the discrete energy converges more slowly than spectral quadrature, so a tighter tolerance would fail for the wrong reason). On a network output, a disagreement above 2% with C1 passing points to a region where the mesh is folded or the triangles are badly shaped; a disagreement with C1 failing is the same defect seen twice.

**Order of attempt.** C8, then C7, then C6, following the plan's week 8 order; C6 is cheapest but needs the checkpoint series from a fresh run.

## 10. Resolution protocol: Gauss-Bonnet first, then W

A certified number is only as good as the grid it was computed on. The protocol decides the grid before any pass or fail is read.

1. Compute ∫K dA and W on grids of 48², 96², 192², 384² in that order (resolution\_study).
2. Stop at the first grid where |∫K dA − 2πχ| is below 10⁻⁶ and W changed by less than 10⁻⁶ relative to the previous grid.
3. Report every certificate on that grid, and report the grid size with it.
4. If 384² does not satisfy step 2, the surface is declared unresolved at this cost. Its certified values are reported as bounds with the residual beside them, never as passes.

Why Gauss-Bonnet leads. The bubble experiment in section 4 shows the order in which quantities converge: the topological integral fails visibly (−1.69 at 128²) while W is still moving by tens of units, and it reaches 10⁻⁹ on the same grids where W stops moving. Because the reference value of ∫K dA is known exactly and W's is not, the topological residual is the only convergence indicator with a known zero.

Cost. A 384² grid is 147 456 nodes. With chunked vmap in float64 on a laptop CPU, second derivatives take under a minute; the fourth-derivative residual is run on a smaller grid (32² default, 48² at most) because its cost per point is about sixteen times higher and it is a diagnostic, not a certificate with an exact zero.

## 11. Estimator comparison protocol: seeds, bias, ties, Huber

The question is not whether W\_rep equals W\_cert (it never does exactly) but whether the difference is noise or bias, and whether the Huber cap contributes.

**Procedure.** For each surface, run willmore\_mc with 5 seeds, n = 5000 (the published setting), and record W\_rep (uncapped) and W\_Huber. Report mean and standard deviation of each, and the two differences bias\_rep = mean(W\_rep) − W\_cert and bias\_Huber = mean(W\_Huber) − W\_cert.

**Rules.**

| Quantity | Rule | Reading |
| --- | --- | --- |
| standard deviation of W\_rep | reported always | the Monte Carlo noise floor for this surface at n = 5000; on the Clifford torus it is about 0.3, on a torus with a localised feature 0.5 to 1.1 |
| bias\_rep | significant if | bias |
| bias\_Huber − bias\_rep | significant if | difference |
| ties | two surfaces whose W\_rep differ by less than 2 · sd are declared tied | no ranking claim is made between them from W\_rep alone; rank by W\_cert |

**Consequence for reading the training logs.** In the genus 2 reproduction, the train-sample energy moved between 28.9 and 30.6 on consecutive epochs while the evaluation value moved by tenths. That spread is the noise floor above, not descent, and any claim of improvement below one standard deviation is a tie. The same rule applies to the published comparison of 30.19 against the reproduction's 29.70.

**Exact zero for the noise.** The Möbius identity (section 5) gives a second measurement of the same noise: W\_rep(Ψ∘φ) − W\_rep(φ) over seeds has true value zero, so its spread is a calibration of sd that does not depend on knowing W\_cert.

## 12. Attack protocol: success rule, the search ladder, reward

An attack is a surface that the training accepts and a certificate rejects. Both halves are defined precisely so that a success cannot be argued away.

**Accepted by the training** means all of the following, evaluated with the published code on the published settings: the total training loss (Huber energy plus regularity terms with the published weights) is at most the loss of the base surface plus 2 · sd from section 11; the regularity thresholds hold everywhere on a 5000-point sample (area element above 0.01, E and G in \[0.001, 5\], mean area element above 0.2); no adaptive safeguard of the training loop would have fired.

**Rejected by a certificate** means one of: C1 fails on the converged grid; the C3 decision table lands in the contradiction row; or the certified energy exceeds the reported energy by more than 2π (half a quantum), with C1 still passing so that the excess is not a topology change. The third form is the one the bubble produces; the first two are the stronger results.

**Reward.** reward = W\_cert − W\_Huber, computed on the converged grid and the 5-seed mean, subject to the acceptance constraints. This is the plan's reward, made computable.

**The ladder, with what each rung is allowed to touch.**

| Rung | Search space | Budget | Stop when |
| --- | --- | --- | --- |
| 1. Random search | (ε, ρ, u₀, v₀) of one bubble on a fixed base, plus the sampling seed | 4 × 4 grid in (ε, ρ), 3 centres, 5 seeds | a point with reward above 2π and C1 passing is found, or the grid is exhausted |
| 2. Greedy local search | the same four parameters, one step at a time, keeping improvements | 200 evaluations | 20 steps without improvement |
| 3. Simulated annealing | small perturbations of the trained network's weights (last two layers, relative size 10⁻³ to 10⁻²) | 2000 evaluations | acceptance rate below 5% for 200 steps |
| 4. Tabular Q-learning | discretised (ε, ρ, centre) actions | only if rung 3 stalls | the plan allows stopping before this rung |

Rung 3 is the interesting one for the audit: it asks whether the flow's own parameter space contains surfaces near the trained one that fool the loss, which is a statement about the published method, not about hand-made bubbles.

**Initial data search.** The second half of Problem (b): from which initial tori does the published flow itself produce a rejected surface. Procedure: take the perturbed initial surfaces of the basin map (section 13), run the published flow unchanged, and pass every final checkpoint through the audit. A rejected output here is a stronger finding than any grafted bubble, because nothing was grafted.

## 13. Deliverables, specified column by column

**The audit table.** One row per surface, one column per certificate. Every cell carries the value and the verdict, never the verdict alone.

| Column | Content | Format |
| --- | --- | --- |
| Surface | name, genus, source (exact, checkpoint run and epoch, attack rung) | text |
| Grid | grid size from the resolution protocol, converged yes or no | 96², yes |
| W\_cert | certified energy and ratio to the known minimum | 19.7392 (1.0000×) |
| W\_rep ± sd | 5-seed mean and standard deviation of the training estimator; Huber value beside it when it differs | 19.82 ± 0.31 |
| C1 | χ̂ and residual, pass or fail | χ̂ = 0.000, 4.6e-11, pass |
| C2 | relative difference after inversion, pass or fail | 6e-9, pass |
| C3 | multiplicity bound, embedded flag, mesh result when available | ≤ 1, embedded |
| C4 | relative residual and band | 1.6e-6, critical |
| C8 | Bobenko energy and disagreement with W\_cert, when available | pending |
| Attack | best reward found against this surface and the rung that found it | pending |

Rows planned for the presentation: exact sphere, exact Clifford, exact torus R = 3, genus 0 checkpoint, genus 1 checkpoint (run\_1, W\_rep = 19.74), genus 2 reproduction (run\_9 and run\_11, W\_rep = 29.70 and 29.23, certificates C1 to C4 only once the two-chart quadrature exists), and the four bubble rows of section 8.

**Figure 1. Certified against reported energy.** x axis W\_cert, y axis W\_rep with error bars from the seeds, one marker per surface, a second marker series for the Huber value, the diagonal as reference, and a vertical line at the known minimum. Points on the diagonal are surfaces the estimator handles; points below it are surfaces the training under-reports; the distance from the diagonal is the reward of section 12. attack\_bump.py produces it.

**Figure 2. The basin map (week 7).** Final surfaces of the perturbed-initialisation runs, embedded by three Möbius-invariant coordinates (W\_cert, the L² norm of the trace-free second fundamental form, the fitted R/r), clustered in the quotient by the symmetry group, each point annotated with its C4 band and, where available, its C7 index. It answers Problem (c): which basins the flow reaches, and whether a stall is numerical or a non-minimal Willmore surface.

**The ingredient table (Problem (d)).** Rows are the alternative methods, each replacing one ingredient of the published method (ansatz, curvature computation, loss, flow); columns are the certificates. A method that keeps a certificate the published method loses, or loses one it keeps, says which ingredient that guarantee depends on. Week 8, in the plan's order: linear spectral ansatz, residual-loss PINN, optimiser arm (Adam, L-BFGS, Sobolev), then Bobenko, then Hessian index.

**Repository.** willmore-audit in the course format: certificates.py, the three scripts, tests that assert the exact-surface values, the numpy validation with its log, and a results folder with one JSON per audited surface.

## 14. Schedule to the presentation and the cutting rule

The presentation is in the week of 4 October. What goes on a slide is only what has run. The cutting rule: if a step is not finished by its date, it moves to the next-steps slide and is named there as scoped out, not as failed.

| Date | Step | Output |
| --- | --- | --- |
| Oct 1, 2026 | pytest on the exact surfaces in torch; audit\_checkpoint.py on the genus 1 checkpoint (run\_1) with the resolution study | audit\_run1.json; rows 1 to 3 and 5 of the audit table |
| Oct 2, 2026 | audit of the genus 0 checkpoint; attack\_bump.py on Clifford and on run\_1; 5 seeds | Figure 1 with real points; the bubble rows |
| Oct 3, 2026 | C4 on both checkpoints; fallback if noisy; slides frozen with the numbers in hand | the deck |
| Oct 4, 2026 | rehearsal against the question list in section 15 | timing under 15 minutes |

After the presentation, in the plan's order: two-chart quadrature for genus 2 (a grid per torus chart with the disc removed and the collar counted once), mesh self-intersection test, C8, perturbed-initialisation runs and the basin map, annealing rung, C7.

**Not claimed at any point.** Anything about the genus 2 minimiser; optimality of any method in general; that a certificate passing proves the surface is the minimiser. A pass says the output is consistent with a Willmore surface of that genus at that resolution, nothing more.

## 15. Questions to expect, with answers

The lecturers are the authors of the audited paper, so the questions will be precise. Each answer below is one to three sentences and points to the section that backs it.

**Your quadrature is also an estimator. Why trust it over Monte Carlo?** Because it has an exact zero to check itself against. Gauss-Bonnet on the same grid returns 10⁻⁹ on the exact surfaces and the resolution protocol refuses any grid where it does not close (sections 4 and 10). Monte Carlo has no such internal check; its only reference is the quadrature.

**The Huber cap does not change the minimiser. Why do you count it as a defect?** It does not change where the minimum is; it changes what the optimiser sees on the way there. A bubble with |H| above 7 is charged linearly instead of quadratically, so a surface with 38.8 uncapped energy reports 25.9 to the optimiser (section 8). The cap is a defensible training choice and an indefensible reporting choice; the audit separates the two.

**A bubble is artificial. Does the flow ever produce one?** That is Problem (b), second half, and it is week 7 (section 12). The bubble shows what the estimator cannot see; the perturbed-initialisation runs test whether the flow walks into that blind spot on its own. The claim on the slide is only the first part.

**Fourth derivatives of a tanh network by autodiff: are they meaningful?** On the exact surfaces the residual is 10⁻⁶, so the operator is correct. On a network the residual mixes the geometry with the network's own roughness; that is why the threshold is a band, not a line, and why the fallback through the first variation of W exists (section 7). The residual is reported as a number with its band, not as a pass.

**Why is genus 2 missing from the certificates?** Two glued charts need a quadrature that removes the disc from each chart and counts the collar once; that is not a conceptual obstacle but it is not written yet (section 14). Everything said about genus 2 here rests on the reported energy and on the warm-restart evidence that 29.7 is a local minimum, which is an estimator-level claim.

**Why 5 seeds and n = 5000?** Because that is the published setting; the point is to measure the published estimator as used, not a better one. The standard deviation from 5 seeds is a rough number, and the ties rule uses 2 sd to absorb that (section 11).

**Your Li-Yau test only bounds multiplicity. Is that a certificate?** For genus 1 it is a complete one: a certified energy below 8π forces embeddedness. For genus 2 at 29.7 it says nothing, which is stated as such. The mesh test that would speak for genus 2 is week 7 (section 6).

**How does this differ from just computing W on a finer grid?** A finer grid gives a better number; it does not give a reason to believe it. The certificates give reasons with exact reference values, and the C3 contradiction row can convict the estimator outright, which no amount of refinement can do.

**What in the course does this use?** The certificates are analysis made computable by autodiff (Lectures 4, 12, 13 for the PINN machinery and geometric flows). The attacks are the search ladder of Lecture 9. The basin map is Lectures 7 and 8. The alternative ansätze are Lectures 3, 5, 6, 10, 11. The Plateau demo is the mesh-flow baseline.

## 16. Sources

Numbering follows the project plan (project\_plan\_nina.pdf).

- &#91;1\] T. J. Willmore. Note on embedded surfaces. An. Şti. Univ. Iaşi 11B (1965).
- &#91;2\] W. Blaschke. Vorlesungen über Differentialgeometrie III. Springer, 1929.
- &#91;3\] J. H. White. A global invariant of conformal mappings in space. Proc. AMS 38 (1973).
- &#91;4\] P. Li, S.-T. Yau. A new conformal invariant and its applications to the Willmore conjecture. Invent. Math. 69 (1982).
- &#91;5\] F. C. Marques, A. Neves. Min-max theory and the Willmore conjecture. Ann. of Math. 179 (2014).
- &#91;7\] L. Simon. Existence of surfaces minimizing the Willmore functional. Comm. Anal. Geom. 1 (1993).
- &#91;8\] T. Rivière. Analysis aspects of Willmore surfaces. Invent. Math. 174 (2008).
- &#91;9\] E. Kuwert, R. Schätzle. Gradient flow for the Willmore functional. Comm. Anal. Geom. 10 (2002).
- &#91;11\] J. L. Weiner. On a problem of Chen, Willmore, et al. Indiana Univ. Math. J. 27 (1978).
- &#91;13\] A. I. Bobenko. A conformal energy for simplicial surfaces. MSRI Publ. 52 (2005).
- &#91;15\] E. Hirst, H. N. Sá Earp, T. S. R. Silva. Minimising Willmore energy via neural flow. [arXiv:2604.04321](https://arxiv.org/abs/2604.04321) (2026). Code: [WillmorePINN](https://github.com/edhirst/WillmorePINN).
- &#91;18\] E. Hirst, T. S. R. Silva. MM845: AI for Geometry, lecture slides and tutorials. [github.com/TomasSilva/MM845](https://github.com/TomasSilva/MM845).
- Harness and validation log: willmore-audit (certificates.py, validate\_numpy.py, validation\_numpy\_log.txt), this project.
- Experiment log of the reproduction: EXPERIMENTOS\_willmore\_sobolev.md, willmorepinn-sobolev repository, this project.
