# kappa-kinisi

Numerical stability of covariance-matrix inversions in `kinisi`.

## Problem
`kinisi` fits a diffusion coefficient *D* to the MSD, weighting by the full covariance matrix Σ.
Σ is often non-positive-definite, and in rare cases its inversion returns a badly wrong *D* while
standard diagnostics report the matrix as healthy.


<img src="bp2.png" alt="alt text" width="200">


## Results (all reproduced in the notebooks)
- **Rare & real:** 64,000 random-walk fits (16/24/32/48 atoms) -> 14 anomalies (~0.02%).
- **No diagnostic separates them:** condition number, κ_pos, negative fraction, spectral deviation
  all fail. κ_pos looked perfect but is a scoring artifact (796 normals above the weakest anomaly;
  fails within-group and out-of-sample).
- **Magnitudes are blind:** normal seed 0 has λ_min = −6802, more negative than all broken seeds.
- **Adaptive eigenvalue floor** (lift eigenvalues below `c·|λ_min|`, c = 0.25) repairs matrix-driven
  failures on the *real* Bayesian fit: broken seeds 0.000 -> ~1; normal cases untouched; coverage
  near-ideal (1000 seeds) at ~2× wider intervals.
- **Origin:** finite-sampling noise in the long-time variances (negative fraction 0.905 -> 0.005 as
  atoms 64 -> 16,384; analytical matrix is always positive-definite).

## Caveats
Fixed `mineig` (κ=1e3) repairs about as well - the floor's edge is being parameter-free, robust
across inverses, and honestly calibrated, not a better point estimate. n=14 supports no predictive
claim. Proxies disagree with the real fit - trust the real fit. LiPS numbers (1.1e-8 -> 4e-6) are a
motivating claim, not yet reproduced.

## Layout
```
branch/ main, basic-testing, suspect_a_testing, suspect_b_testing

code/    Anomalous_B_Value_Inspect_full (core), flooring, Suspect_A/B_Testing,
         method_comparison, model-nodel
data/    .npy/.npz (untracked)   plots/   LOG.md
```

## Environment
Python 3.11, conda env `kappa-kinisi`, `kinisi` 2.0.5 (+ MDAnalysis, scipp, numpy, scipy, joblib).
Long runs: tmux + joblib, checkpointed to `.npy`.

## Reproduce
Run `Anomalous_B_Value_Inspect_full.ipynb` (population + diagnostics + real-fit recovery), then
`flooring.ipynb` (c-sweep, coverage) and `Suspect_A/B_Testing.ipynb` (origin).

## Notes
True D = 1 (kinisi reports ~9,980). Raw Σ must be rebuilt via `build_raw_cov`; inject treatments by
monkey-patching `a.diff.compute_covariance_matrix`.

## Reference
McCluskey, Coles & Morgan. *Accurate Estimation of Diffusion Coefficients and their Uncertainties
from Computer Simulation.* J. Chem. Theory Comput. 2025, 21, 79–87.