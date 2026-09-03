"""
Determine the coefficient of the adaptive eigenvalue floor.

Produces Table 5.3 of the dissertation. The floor lifts any eigenvalue below
c|lambda_min| to that value, and c is swept across two orders of magnitude to
establish whether the choice matters and on what grounds it should be made.

The result is that recovery is essentially flat across the range, so accuracy
cannot select a value, while the width of the posterior grows monotonically
with c. The coefficient adopted, c = 0.25, is accordingly the smallest which
recovers every affected case, and lies near the narrow end of the width curve.

Outputs
    ~/data/anomaly_realfit_c_sweep.npy    one record per (case, c)

Usage
    python realfit_c_sweep.py

"""


import os
import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer
from pathlib import Path
from tqdm import tqdm


base = Path.home()



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

c_values = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]

ckpt = base / "data" / "anomaly_realfit_c_sweep.npy"

if os.path.exists(ckpt):
    rows = list(np.load(ckpt, allow_pickle=True))
    done = {(r["atoms"], r["seed"], r["c"]) for r in rows}
    print(f"resuming - {len(rows)} fits already done")
else:
    rows, done = [], set()

for atoms, seed in tqdm(bad):
    for c in c_values:
        if (atoms, seed, c) in done:
            continue
        try:
            post = realfit_with_treatment(seed, lambda m, c=c: adaptive_floor(m, c=c), atoms=atoms)
            rows.append({"atoms": atoms, "seed": seed, "c": c,
                         "d": float(np.median(post)), "d_std": float(np.std(post))})
        except Exception as e:
            rows.append({"atoms": atoms, "seed": seed, "c": c, "d": np.nan, "d_std": np.nan,
                         "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt, np.array(rows, dtype=object))

hdr = "  ".join(f"c={c:<5}" for c in c_values)

print(f"\n{'atoms':>5} {'seed':>6}  {hdr}")

for atoms, seed in bad:
    g = {r["c"]: r for r in rows if r["atoms"] == atoms and r["seed"] == seed}
    vals = "  ".join(f"{g[c]['d']:7.3f}" if c in g and np.isfinite(g[c]["d"]) else f"{'nan':>7}"
                     for c in c_values)
    print(f"{atoms:5d} {seed:6d}  {vals}")

print("\nmedian recovered D per c:")


for c in c_values:
    ds = np.array([r["d"] for r in rows if r["c"] == c and np.isfinite(r["d"])])
    frac = np.mean((ds > 0.5) & (ds < 2.0)) if ds.size else np.nan
    print(f"  c={c:<5}  median D = {np.median(ds):6.3f}   recovered (0.5-2.0): {frac:4.0%}")

errs = [r for r in rows if "error" in r]

if errs:
    print("\nerrors:")
    for r in errs:
        print(f"  {r['atoms']} {r['seed']} c={r['c']} -> {r['error']}")
