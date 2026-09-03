"""
Compare the four corrections against one another upon a common footing.

Produces Table 5.4 of the dissertation. Each of the fourteen affected cases and
four healthy controls is refitted under four treatments: untreated, the
adaptive floor, shrinkage, and shrinkage followed by the floor.

The final combination answers a specific question. Shrinkage does not eliminate
the indefiniteness, reducing the number of negative eigenvalues from a median
of 21 to a median of 4, and the affected fits recover because the residual
damage no longer dominates the inverse rather than because the matrix has been
made valid. Applying the floor afterwards removes those four and is
nonetheless inferior, which establishes that positive definiteness is not the
property governing the quality of the result.

The number of negative eigenvalues remaining after each treatment is recorded
alongside the recovered coefficient, since that is the quantity the argument
turns upon.

Outputs
    ~/data/shrink_floor_test.npy    one record per (case, treatment), including
                                    the count of negative eigenvalues remaining

Usage
    python shrink_floor_test.py
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
length = 128



def walk(atoms, timesteps, jump_size, rng):
    """
    A three-dimensional lattice random walk.

    Each particle steps one jump along one of six axis directions at every
    timestep. With a jump of sqrt(6) the true diffusion coefficient is exactly
    unity, since D = kappa^2 / (2d), which is what makes a failed fit
    identifiable at all.

    """
    moves = np.zeros((6, 3))
    axis = 0
    for i in range(0, 6, 2):
        moves[i, axis] = jump_size
        moves[i + 1, axis] = -jump_size
        axis += 1
    return np.cumsum(moves[rng.choice(6, size=(atoms, timesteps))], axis=1)






def build_analyzer(seed, atoms=32):
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

    a = DiffusionAnalyzer.from_universe(
        u, time_step=1.0 * sc.Unit('s'), step_skip=1,
        distance_unit=sc.Unit('m'), specie='A',
        dt=sc.linspace(dim='time interval', start=2 * sc.Unit('s'),
                       stop=length * sc.Unit('s'), num=126),
        progress=False)
    
    a.diffusion(2 * sc.Unit('s'), progress=False)
    return a






def raw_pieces(a):
    """
    Return the variances, sample counts and lag times of the diffusive regime.

    These are the 126 measured quantities from which the covariance is generated,
    and the objects the variance treatments operate upon.

    """
    da = a.diff.dg['da']

    regime = a.diff.diff_regime

    return (da.data.variances[regime:],
            da.coords['n_samples'].values[regime:],
            da.coords['time interval'].values[regime:])






def cov_from_var(var, nprime):
    """
    Assemble the covariance from a variance profile.

    Applies the generating formula: entry (i, j) is the variance at the shorter
    lag scaled by the ratio of the two sample counts.

    """
    n = var.size
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = (nprime[i] / nprime[j]) * var[i]
            cov[j, i] = cov[i, j]
    return cov






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
    v, vecs = np.linalg.eigh(m)
    if v[0] < 0:
        floor = c * (-v[0])
        v = np.where(v < floor, floor, v)
    return (vecs * v) @ vecs.T





def fit_aprime(var, nprime, nsteps):
    """
    Fit the amplitude of the analytical variance form to the data.

    Weighted by sample count so the well-determined short lags dominate. Fitting
    rather than assuming the amplitude is what allows the treatment to
    generalise to a system whose true diffusion coefficient differs.

    """
    g = nsteps ** 2 / nprime
    return float(np.sum(nprime * var * g) / np.sum(nprime * g * g))





def shrunk_var(var, nprime, nsteps, tau_factor=1.0):
    """
    Blend each variance toward the analytical form by its sample count.
    """
    ah = fit_aprime(var, nprime, nsteps)
    model = ah * nsteps ** 2 / nprime
    tau = tau_factor * float(np.median(nprime[len(nprime) // 2:]))
    w = nprime / (nprime + tau)
    return w * var + (1 - w) * model





def realfit_with_cov(a, treated, start=2.0):
    """
    Refit with a treated covariance, through the genuine pipeline.

    Injected by monkey-patching because bayesian_regression recomputes the
    covariance internally and would otherwise discard it.
    """
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
        return a.diff.gradient.values / (2 * 3)
    finally:
        a.diff.compute_covariance_matrix = original



def negatives(cov):
    """
    Count the eigenvalues below zero.

    Recorded alongside each treatment because the argument turns on it: a matrix
    retaining four negative eigenvalues gives a better fit than the same matrix
    with none.

    """
    ev = np.linalg.eigvalsh(cov)
    return int((ev < 0).sum()), float(ev.min())




bad14 = [tuple(map(int, b)) for b in np.load(base / "data" / "all_bad_seeds_v2.npy")]

controls = [(16, 0), (24, 0), (32, 0), (48, 0)]

targets = bad14 + controls


ckpt = base / "data" / "shrink_floor_test.npy"

if os.path.exists(ckpt):
    rows = list(np.load(ckpt, allow_pickle=True))

    done = {(r["atoms"], r["seed"], r["method"]) for r in rows}

    print(f"resuming - {len(rows)} fits done")
else:
    rows, done = [], set()



methods = ["raw", "floor", "shrink", "shrink_floor"]


for atoms, seed in tqdm(targets):
    todo = [m for m in methods if (atoms, seed, m) not in done]
    if not todo:
        continue

    a = build_analyzer(seed, atoms=atoms)

    var, nprime, nsteps = raw_pieces(a)

    sv = shrunk_var(var, nprime, nsteps)


    built = {"raw": cov_from_var(var, nprime),
             "floor": adaptive_floor(cov_from_var(var, nprime)),
             "shrink": cov_from_var(sv, nprime),
             "shrink_floor": adaptive_floor(cov_from_var(sv, nprime))}
    

    for name in todo:
        cov = built[name]
        n_neg, lmin = negatives(cov)
        try:
            post = realfit_with_cov(a, cov)
            rows.append({"atoms": atoms, "seed": seed, "method": name,
                         "d": float(np.median(post)), "d_std": float(np.std(post)),
                         "n_neg": n_neg, "lmin": lmin})
        except Exception as e:
            rows.append({"atoms": atoms, "seed": seed, "method": name,
                         "d": np.nan, "d_std": np.nan, "n_neg": n_neg, "lmin": lmin,
                         "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt, np.array(rows, dtype=object))



print("\nrecovered D (true = 1)")
print(f"{'atoms':>5} {'seed':>6} " + " ".join(f"{m:>13}" for m in methods))
for atoms, seed in targets:
    g = {r["method"]: r for r in rows if r["atoms"] == atoms and r["seed"] == seed}
    vals = " ".join(f"{g[m]['d']:13.3f}" if m in g and np.isfinite(g[m]["d"])
                    else f"{'nan':>13}" for m in methods)
    print(f"{atoms:5d} {seed:6d} {vals}  {'CONTROL' if seed == 0 else ''}")



print("\nnegative eigenvalues remaining")
print(f"{'atoms':>5} {'seed':>6} " + " ".join(f"{m:>13}" for m in methods))
for atoms, seed in targets:
    g = {r["method"]: r for r in rows if r["atoms"] == atoms and r["seed"] == seed}
    vals = " ".join(f"{g[m]['n_neg']:13d}" if m in g else f"{'-':>13}" for m in methods)
    print(f"{atoms:5d} {seed:6d} {vals}  {'CONTROL' if seed == 0 else ''}")



print("\nsummary over the 14 anomalies")
badset = set(bad14)
for m in methods:
    d = np.array([r["d"] for r in rows if r["method"] == m
                  and (r["atoms"], r["seed"]) in badset and np.isfinite(r["d"])])
    sd = np.array([r["d_std"] for r in rows if r["method"] == m
                   and (r["atoms"], r["seed"]) in badset and np.isfinite(r["d_std"])])
    ng = np.array([r["n_neg"] for r in rows if r["method"] == m
                   and (r["atoms"], r["seed"]) in badset])
    rec = np.mean((d > 0.5) & (d < 2.0)) if d.size else np.nan
    print(f"  {m:13s} median D {np.median(d):6.3f}   median d_std {np.median(sd):7.4f}"
          f"   recovered {rec:4.0%}   median negatives {int(np.median(ng)):3d}")



print("\ncontrols (should stay near 1)")
for m in methods:
    d = np.array([r["d"] for r in rows if r["method"] == m
                  and r["seed"] == 0 and np.isfinite(r["d"])])
    print(f"  {m:13s} " + "  ".join(f"{x:.3f}" for x in d))

errs = [r for r in rows if "error" in r]
if errs:
    print(f"\nerrors: {len(errs)}")
    for r in errs[:5]:
        print(" ", r)
