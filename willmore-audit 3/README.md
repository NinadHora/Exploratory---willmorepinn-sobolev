# willmore-audit

Certificates and adversarial tests for the neural Willmore flow of Hirst, Sá Earp and Silva
([arXiv:2604.04321](https://arxiv.org/abs/2604.04321), code: [WillmorePINN](https://github.com/edhirst/WillmorePINN)).
Mini-project for MM845 "AI for Geometry" (IMECC, Unicamp, 2026).

The harness is independent of the training loop. It takes **any** differentiable map
φ:(u,v)→R³ in torch (a WillmorePINN checkpoint, an exact formula, a perturbed surface)
and checks identities that every genuine closed surface must satisfy, on a grid the
training never saw. The training never sees the harness; the harness never sees the loss.

Two numbers are kept apart throughout:

- **reported energy** W_rep: what the training computes and optimises (Monte Carlo on 5000
  uniform points, Huber cap on H² at 50);
- **certified energy** W_cert: spectral quadrature in float64 on a grid refined until
  Gauss–Bonnet closes.

| Certificate | What it checks | Function |
|---|---|---|
| C1 Gauss–Bonnet | ∫K dA = 2πχ = 4π(1−g) | `gauss_bonnet` |
| C2 Möbius | W[Ψ∘φ] = W[φ] for two inversions (centre outside and at the centroid) | `mobius_test` |
| C3 Li–Yau | W < 8π ⇒ embedded (multiplicity ≤ ⌊W/4π⌋); mesh self-intersection test | `li_yau`, `mesh_check.py` |
| C4 Willmore equation | ΔH + 2H(H²−K) = 0 at critical points (4th derivatives by autodiff), three bands: critical < 0.05 < near-critical < 0.5 < not critical | `willmore_residual` |
| C4 fallback | first variation of W under normal perturbations (2nd derivatives only); identity dW/dε = ∫ r ξ dA validated | `willmore_residual_variational` |
| C5 bounds | W ≥ 4π (g=0), W ≥ 2π² (g=1) | in `certify` |
| estimator | replica of the training estimator (uniform MC + Huber), 5 seeds: bias, significance, Huber active | `willmore_mc`, `estimator_report` |
| resolution | refine the grid until W and Gauss–Bonnet close | `resolution_study` |
| attack | grafted bubble φ + ε·exp(−d²/ρ²)·n; reward = W_cert − W_Huber | `graft_bump`, `attack_bump.py` |

The thresholds, the pass rules and what each failure means are fixed in
[`docs/protocol.md`](docs/protocol.md), written before the checkpoints were audited.

## Layout

```
certificates.py            the harness (9 blocks); the only file that computes a certificate
audit_checkpoint.py        all certificates on one WillmorePINN checkpoint (--resolution-study --variational)
attack_bump.py             bubble attack on a trained surface + Figure 1 (certified vs reported)
diag_H.py                  H and K maps of a surface, spectral energy split (used for the C4 reading)
mesh_check.py              C3, second half: triangle mesh of the output, self-intersection test (numpy, Möller 1997)
trajectory_map.py          one training run as a curve: W_cert per epoch (C6), PCA of the H maps, high-mode energy per epoch
validate_numpy.py          numpy mirror (4th-order finite differences): independent validation of the maths
make_table.py              audit table (markdown + csv) from the JSONs in runs/
run_registry.py            audit / train / attack / basins; every output becomes runs/<kind>__<label>__<commit>.json
run_tonight.sh             the overnight sequence used for the results below
tests/test_certificates.py pytest on the exact surfaces (18 tests)
methods/                   each strategy of the plan becomes a Surface (phi, genus, meta)
  base.py                  Surface + REGISTRY + Surface.audit()
  published.py             the published flow: from_checkpoint, train(config overrides)
  linear_spectral.py       linear Fourier ansatz (no hidden layers), ran
  optimisers.py            adam / adamw / sgd via config, ran; sobolev needs the external patch; lbfgs to write
  pinn_residual.py         residual-loss PINN, weak form (first variations); ran
  others.py                cnn_periodic, set_transformer, mesh_flow: interface only
attacks/ladder.py          rungs 1 and 2 (random search, greedy) implemented; annealing, Q-learning skeleton
basins/
  perturb_init.py          Fourier-perturbed tori, twisted τ, (2,3) knot tube as initial data
  basin_map.py             Möbius-invariant coordinates, DBSCAN, kernel PCA, Figure 2
  run_local.py             reduced basin map for one night on a laptop (9 initial data)
slurm/                     env_setup.sh, audit / attack / train_arm / basins .sbatch for a SLURM cluster
environment.yml            conda env (torch ≥ 2.2 for torch.func)
docs/                      protocol.md, roteiro.md (talk script, Portuguese), slides (pptx, pdf), build_slides.js, qr/
results/                   audit table, attack csv, Figure 1, H maps, numpy validation log
runs/                      JSON per audited surface (add your own; see "Reproducing")
```

Rule: no method calls `certificates.py` directly; everything goes through `Surface.audit()`.
That is what makes the rows of the audit table comparable.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy scipy matplotlib pyyaml h5py tqdm pytest scikit-learn

# 1. unit tests on the exact surfaces (sphere, ellipsoid, Clifford, torus R=3): ~5 s
python -m pytest tests -v

# 2. audit a checkpoint of the published flow (genus 0 or 1)
python audit_checkpoint.py --repo ~/WillmorePINN-main \
    --ckpt ~/WillmorePINN-main/checkpoints/run_1/best_model.pt \
    --resolution-study --variational --out audit_run1.json

# 3. bubble attack on the trained torus (16 bubbles, 5 seeds each), writes fig1
python attack_bump.py --repo ~/WillmorePINN-main --ckpt .../run_1/best_model.pt --outdir results_run1

# 4. everything else, in sequence (alternative methods, extra seeds, basins, table)
bash run_tonight.sh ~/WillmorePINN-main
python make_table.py

# 5. independent validation without torch
python validate_numpy.py
```

## Results (1 October 2026, laptop CPU, float64)

Full table in [`results/audit_table.md`](results/audit_table.md); mesh results in `runs/mesh__*.json`. Headline numbers:

- Exact surfaces: all certificates close to machine precision (W = 4π, 2π², π²R²/(r√(R²−r²));
  GB residual ~1e-15; Möbius ~1e-16; C4 ~1e-6 on sphere and Clifford, 0.81 on the torus R=3).
- Genus 1 checkpoint (run_1, epoch 141): W_cert = 19.9710 (1.0117 × 2π²), grid 192², GB 2.7e-15,
  Möbius 1.8e-16 / 5.3e-16, embedded by Li–Yau. C4 direct 0.99 (not critical), variational 0.21
  (near-critical): correct at large scale, fine ripples above the feature band (98.1 % of the
  spectral energy of H sits in modes ≤ 6). W_rep = 19.906 ± 0.092 over 5 seeds; saved loss 19.721.
- Genus 0 checkpoint (run_2): the last epoch is better than the saved "best": W_cert = 12.5981
  (1.0025 × 4π) versus 12.6997 (1.0106 ×) for the checkpoint selected by reported energy, whose
  saved loss (12.424) is below 4π, which no sphere can have. C4 0.94 / 0.22: smooth but not round.
- Bubble attack on run_1: the uncapped estimator is unbiased but its spread grows as the bubble
  narrows; the Huber value the optimiser descends saturates. A bubble of height 0.05 and width
  0.15 takes W_cert to 29.3 (> 8π) while the optimiser sees 23.9 (< 8π): the training signal
  crossed the Li–Yau line. Up to 24 % of the energy hidden; topology certified on those rows.
- Alternatives and initial data (overnight): linear spectral ansatz 1.0033 ×; published flow
  with seeds 1, 2: 1.0070 ×, 1.0084 ×; from Fourier-perturbed Clifford-like tori 1.0014 to 1.0021 ×
  (all six saved losses below 2π²); twisted τ 1.0046 and 1.0176 ×; from a (2,3) knotted tube the
  flow stalls at 1.88 × (37.1 > 8π). Initial data matters about ten times more than the sampling
  seed; depth is worth about 0.15 % on genus 1.
- Mesh self-intersection test (`mesh_check.py`, 96² grid): exact sphere, Clifford, torus R=3, run_1,
  run_2, the two twisted-τ outputs, the bubble (0.05, 0.15) and the knotted initial tube: no crossing.
  The 0.3-high bubble crosses the opposite wall of the hole (74 pairs) and the training's regularity
  thresholds accept it. The torus trained from the knotted tube crosses itself (784 pairs): the flow
  unknotted an embedded tube by passing through itself and stalled on an immersed torus; the loss has
  no self-intersection term, Li–Yau is silent above 8π, only the mesh sees it.
- Training trajectory (`trajectory_map.py`, run_1, 201 per-epoch checkpoints): the certified energy
  rises in 58 of 200 epochs (C6, Kuwert–Schätzle: the true flow is monotone, the training is not);
  the best-by-reported checkpoint (epoch 141, 1.0117 ×) is 0.5 % worse than epoch 199 (1.0068 ×);
  2 principal components of the H maps carry 90 % of the motion, 11 carry 99 %; the spectral energy
  of H above mode 6 goes 32 % → 2 % (epoch 40) → 1 % (epoch 200). Figure: `results/fig3_trajectory_run1.png`.
- Residual-loss PINN, weak form (`methods/pinn_residual.py`): the loss is the sum of squared first
  variations of W along ξ_k e_i (6 test functions × 3 axes, central differences, second derivatives only),
  plus 0.05 W and the area floor; lr 1e-4 with gradient clipping (lr 1e-3 diverged). It drives its own
  residual from 35 to 0.05 and ends at W_cert = 20.9631 (1.0620 ×), grid 192², GB 4e-15, Möbius 2e-16,
  embedded, direct C4 0.98 (not critical): critical only in the directions the loss tested. The strong
  form (pointwise ΔH + 2H(H²−K), fourth derivatives) does not run: reverse mode over four nested
  forward derivatives under `vmap` raises an in-place error in the torch build used.

Not claimed: anything about the genus 2 minimiser (the two-chart quadrature is not written);
optimality of any method in general; that a passing certificate proves a surface is the minimiser.

## Next steps, in the plan's order

Two-chart quadrature for genus 2; full basin map on the
cluster (`slurm/basins.sbatch`); annealing rung of the attack; Sobolev and L-BFGS arms; Bobenko's
discrete energy (C8); Hessian index (C7); fix `pinn_residual`.

## Tooling

The harness code was drafted with an AI assistant (Claude) from my specification; the protocol
text and slides were organised with its help. The certificates, thresholds, experimental
decisions, all runs and the interpretation are mine.

## Reproducing the table

`runs/` is where `run_registry.py` and `basins/run_local.py` write one JSON per audited surface,
and `make_table.py` reads them. The JSONs behind `results/audit_table.md` were produced on the
author's laptop and are not all in this repository; rerun `run_tonight.sh` (about two hours on
an Apple M-series CPU) to regenerate them.
