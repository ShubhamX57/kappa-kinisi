"""
Evaluate corrections applied to the assembled covariance, upon the real fit.

Produces the results reported in Section 5.7.1 of the dissertation. Each of the
fourteen affected cases is refitted under three treatments: untreated, the
adaptive eigenvalue floor at c = 0.25, and a minimum-eigenvalue floor at a
target condition number of 1e3.

The treated matrix is supplied to the genuine Bayesian pipeline rather than to
a least-squares proxy. This matters: bayesian_regression recomputes the
covariance internally, so a matrix passed in by any other route is silently
discarded and the test measures nothing. The injection is performed by
monkey-patching compute_covariance_matrix, and the fidelity of that path is
verified separately in reconstruction.py.

Outputs
    ~/data/anomaly_realfit_all14_v2.npy   one record per (case, treatment):
                                          recovered D, posterior width, and the
                                          ordinary least squares slope

Usage
    python run_anomaly_harness.py

"""


import os
import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer
from pathlib import Path
from tqdm import tqdm
from scipy.stats import linregress



base = Path.home()
(base / "data").mkdir(parents=True, exist_ok=True)




def walk(atoms, timesteps, jump_size, seed):
    """
    A three-dimensional lattice random walk.

    Each particle steps one jump along one of six axis directions at every
    timestep. With a jump of sqrt(6) the true diffusion coefficient is exactly
    unity, since D = kappa^2 / (2d), which is what makes a failed fit
    identifiable at all.
    """
    possible_moves = np.zeros((6, 3))
    j = 0
    for i in range(0, 6, 2):
        possible_moves[i, j] = jump_size
        possible_moves[i + 1, j] = -jump_size
        j += 1
    choices = seed.choice(6, size=(atoms, timesteps))
    steps = possible_moves[choices]
    return np.cumsum(steps, axis=1)





def build_analyzer(seed, atoms=128, length=128):
    """
    Construct a fitted DiffusionAnalyzer for one seed.

    The fit is performed here rather than by the caller because a.diff does not
    exist until diffusion() has been called.
    """
    rng = np.random.RandomState(seed)
    steps = walk(atoms, length, np.sqrt(6), rng)
    dims = np.tile([200.0, 200.0, 200.0, 90.0, 90.0, 90.0], (steps.shape[1], 1))
    u = mda.Universe.empty(steps.shape[0], trajectory=True)
    u.add_TopologyAttr('name', [f'Atom{k}' for k in range(steps.shape[0])])
    u.add_TopologyAttr('type', ['A'] * steps.shape[0])
    u.trajectory = MemoryReader(np.transpose(steps, (1, 0, 2)), dimensions=dims, delta=1.0)
    a = DiffusionAnalyzer.from_universe(u, time_step=1.0 * sc.Unit('s'), step_skip=1,
                                        distance_unit=sc.Unit('m'), specie='A',
                                        dt=sc.linspace(dim='time interval', start=2 * sc.Unit('s'),
                                        stop=length * sc.Unit('s'), num=126), progress=False)
    a.diffusion(2 * sc.Unit('s'), progress=False)
    return a






def build_raw_cov(a):
    """
    Reconstruct the covariance as it stood before treatment.

    kinisi retains only the matrix that remains after its reconditioning, so the
    matrix that caused the failure must be rebuilt from the stored variances.
    Verified against the stored construction in reconstruction.py.
    """
    da = a.diff.dg['da']
    variances = da.data.variances
    n_samples = da.coords['n_samples'].values
    n = variances.size
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ratio = n_samples[i] / n_samples[j]
            cov[i, j] = ratio * variances[i]
            cov[j, i] = cov[i, j]
    regime = a.diff.diff_regime
    # sliced to the diffusive regime: kinisi fits only beyond diff_regime, and
    # the lag times before it are present in the arrays but unused
    return cov[regime:, regime:]






def adaptive_floor(m, c=0.25):
    """
    Lift every eigenvalue below c|lambda_min| to that value.

    A spectral treatment: eigenvectors are untouched and only magnitudes change.
    It cannot distinguish an eigenvalue small through sampling noise from one
    small because the data genuinely varies little in that direction, and alters
    both alike. The coefficient is justified in realfit_c_sweep.py.
    """
    # eigh, not eig: the matrix is symmetric by construction, so the
    # eigenvalues are real and returned in ascending order
    v, V = np.linalg.eigh(m)
    if v[0] < 0:
        fl = c * (-v[0])
        v = np.where(v < fl, fl, v)
    return (V * v) @ V.T






def mineig_floor(m, kappa=1e3):
    """
    Lift the small eigenvalues to a target condition number.

    The treatment kinisi applies on its default path, reproduced for comparison.
    Note it is a floor, whereas the optional reconditioning is eigenvalue
    clipping against a fitted distribution and a different procedure entirely.
    """
    # eigh, not eig: the matrix is symmetric by construction, so the
    # eigenvalues are real and returned in ascending order
    v, V = np.linalg.eigh(m)
    fl = v[-1] / kappa
    return (V * np.where(v < fl, fl, v)) @ V.T






def realfit_with_treatment(seed, treatment_fn, atoms=32, start=2.0):
    """
    Refit with a treated covariance, through the genuine pipeline.

    Injected by monkey-patching because bayesian_regression recomputes the
    covariance internally: a matrix supplied by any other route is silently
    discarded and the test measures nothing.
    """
    a = build_analyzer(seed, atoms=atoms)
    raw = build_raw_cov(a)
    treated = treatment_fn(raw)
    template = a.diff.compute_covariance_matrix()
    treated_var = sc.array(dims=template.dims, values=treated, unit=template.unit)
    original = a.diff.compute_covariance_matrix
    # bayesian_regression recomputes the covariance internally, so the treated
    # matrix must replace the method rather than be passed as an argument.
    # Without this the fit proceeds on the original and the test measures
    # nothing while appearing to succeed.

    a.diff.compute_covariance_matrix = lambda: treated_var
    try:
        a.diff.bayesian_regression(start * sc.Unit('s'), progress=False)
        D_post = a.diff.gradient.values / (2 * 3)
    finally:
        a.diff.compute_covariance_matrix = original
    return D_post





bad = [tuple(map(int, b)) for b in np.load(base / "data" / "all_bad_seeds_v2.npy")]

ckpt = base / "data" / "anomaly_realfit_all14_v2.npy"



if os.path.exists(ckpt):
    rows = list(np.load(ckpt, allow_pickle=True))
    done = {(r["atoms"], r["seed"], r["method"]) for r in rows}
    print(f"resuming - {len(rows)} fits already done")
else:
    rows, done = [], set()



methods = [("raw", lambda m: m), ("adaptive", adaptive_floor), ("mineig1e3", mineig_floor)]



for atoms, seed in tqdm(bad):
    todo = [(n, f) for n, f in methods if (atoms, seed, n) not in done]
    if not todo:
        continue
    a = build_analyzer(seed, atoms=atoms)
    d_ols = float(linregress(a.dt.values, a.msd.values).slope / (2 * 3))
    for name, fn in todo:
        try:
            post = realfit_with_treatment(seed, fn, atoms=atoms)
            rows.append({"atoms": atoms, "seed": seed, "method": name, "d_ols": d_ols,
                         "d": float(np.median(post)), "d_std": float(np.std(post))})
        except Exception as e:
            rows.append({"atoms": atoms, "seed": seed, "method": name, "d_ols": d_ols,
                         "d": np.nan, "d_std": np.nan, "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt, np.array(rows, dtype=object))

print("\natoms   seed   d_ols      raw  adaptive  mineig1e3")



for atoms, seed in bad:
    g = {r["method"]: r for r in rows if r["atoms"] == atoms and r["seed"] == seed}
    if not g:
        continue
    vals = [g.get(m, {}).get("d", np.nan) for m in ("raw", "adaptive", "mineig1e3")]
    cells = " ".join(f"{v:9.3f}" if np.isfinite(v) else f"{'nan':>9}" for v in vals)
    print(f"{atoms:5d} {seed:6d} {g['raw']['d_ols']:7.3f} {cells}")


errs = [r for r in rows if "error" in r]
if errs:
    print("\nerrors:")
    for r in errs:
        print(f"  {r['atoms']} {r['seed']} {r['method']} -> {r['error']}")