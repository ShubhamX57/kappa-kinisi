"""
Evaluate corrections applied to the variance profile before the matrix is built.

Produces the results reported in Section 5.7.2 of the dissertation. Since the
covariance is generated from 126 variances by a fixed formula, a correction may
be applied to those variances rather than to the assembled matrix. Four
treatments are compared.

    parametric              rebuild the entire profile from the analytical form,
                            with the amplitude fitted to the data
    shrink                  blend each variance toward the analytical form in
                            proportion to the number of samples supporting it
    spectral_continuation   extrapolate the decay of the positive spectrum into
                            the negative ranks, avoiding any appeal to the
                            analytical form
    adaptive_floor          the spectral treatment, for comparison

Spectral continuation reduces all four healthy controls to zero and is
therefore disqualified. It is the third construction to fail in that manner,
and the consistency of the outcome is discussed in Section 5.7.2.

Outputs
    ~/data/variance_repair_test.npy    one record per (case, treatment)

Usage
    python variance_repair_test.py
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




def raw_pieces(a):
    """
    Return the variances, sample counts and lag times of the diffusive regime.

    These are the 126 measured quantities from which the covariance is generated,
    and the objects the variance treatments operate upon.
    """
    da = a.diff.dg['da']
    regime = a.diff.diff_regime
    var = da.data.variances[regime:]
    nprime = da.coords['n_samples'].values[regime:]
    nsteps = da.coords['time interval'].values[regime:]
    return var, nprime, nsteps




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
    v, V = np.linalg.eigh(m)
    if v[0] < 0:
        fl = c * (-v[0])
        v = np.where(v < fl, fl, v)
    return (V * v) @ V.T




def fit_aprime(var, nprime, nsteps):
    """
    Fit the amplitude of the analytical variance form to the data.

    Weighted by sample count so the well-determined short lags dominate. Fitting
    rather than assuming the amplitude is what allows the treatment to
    generalise to a system whose true diffusion coefficient differs.
    """
    g = nsteps**2 / nprime
    w = nprime.astype(float)
    return float(np.sum(w * var * g) / np.sum(w * g * g))




def t_parametric(var, nprime, nsteps):
    """
    Rebuild the entire profile from the analytical form.

    The amplitude is fitted but the observed variances are then discarded
    entirely. Shrinkage is preferred because it retains them where the sampling
    supports them.
    """
    ah = fit_aprime(var, nprime, nsteps)
    return cov_from_var(ah * nsteps**2 / nprime, nprime)


def t_shrink(var, nprime, nsteps):
    """
    Blend each variance toward the analytical form by its sample count.

    The weight is N'/(N' + tau): a variance from thousands of samples passes
    through essentially unaltered, one from a few dozen is drawn most of the way
    to the model. No threshold, nothing discarded
    the sampling decides.
    """
    ah = fit_aprime(var, nprime, nsteps)
    model = ah * nsteps**2 / nprime
    tau = float(np.median(nprime[len(nprime) // 2:]))
    w = nprime / (nprime + tau)
    return cov_from_var(w * var + (1 - w) * model, nprime)




def t_spectral_continuation(m):
    """
    Extrapolate the positive spectrum into the negative ranks.

    Attractive because it appeals to no analytical form, and disqualified
    because it reduces all four healthy controls to zero. The third construction
    here to fail that way
    each assigns small eigenvalues to empirically
    determined directions.
    """
    # eigh, not eig: the matrix is symmetric by construction, so the
    # eigenvalues are real and returned in ascending order
    v, V = np.linalg.eigh(m)
    neg = v < 0
    if not neg.any():
        return m
    pos = np.where(~neg)[0]
    k = max(10, min(30, pos.size // 3))
    idx = pos[:k]
    coef = np.polyfit(idx.astype(float), np.log(v[idx]), 1)
    v2 = v.copy()
    fill = np.exp(np.polyval(coef, np.where(neg)[0].astype(float)))
    v2[neg] = fill
    return (V * v2) @ V.T




def realfit_with_cov(seed, cov_fn, atoms=32, start=2.0):
    """
    Refit with a treated covariance, through the genuine pipeline.

    Injected by monkey-patching because bayesian_regression recomputes the
    covariance internally and would otherwise discard it.
    """
    a = build_analyzer(seed, atoms=atoms)
    var, nprime, nsteps = raw_pieces(a)
    treated = cov_fn(var, nprime, nsteps)
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
controls = [(16, 0), (24, 0), (32, 0), (48, 0)]
targets = bad + controls


treatments = [
    ("raw", lambda v, n, t: cov_from_var(v, n)),
    ("floor", lambda v, n, t: adaptive_floor(cov_from_var(v, n))),
    ("parametric", t_parametric),
    ("shrink", t_shrink),
    ("spec_cont", lambda v, n, t: t_spectral_continuation(cov_from_var(v, n))),
]


ckpt = base / "data" / "variance_repair_test.npy"

if os.path.exists(ckpt):
    rows = list(np.load(ckpt, allow_pickle=True))
    done = {(r["atoms"], r["seed"], r["method"]) for r in rows}
    print(f"resuming - {len(rows)} fits done")
else:
    rows, done = [], set()


for atoms, seed in tqdm(targets):
    for name, fn in treatments:
        if (atoms, seed, name) in done:
            continue
        try:
            post = realfit_with_cov(seed, fn, atoms=atoms)
            rows.append({"atoms": atoms, "seed": seed, "method": name,
                         "d": float(np.median(post)), "d_std": float(np.std(post))})
        except Exception as e:
            rows.append({"atoms": atoms, "seed": seed, "method": name,
                         "d": np.nan, "d_std": np.nan, "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt, np.array(rows, dtype=object))

names = [t[0] for t in treatments]



print("\nrecovered D (true = 1)")

print("atoms   seed " + " ".join(f"{n:>11}" for n in names))

for atoms, seed in targets:
    g = {r["method"]: r for r in rows if r["atoms"] == atoms and r["seed"] == seed}
    vals = " ".join(f"{g[n]['d']:11.3f}" if n in g and np.isfinite(g[n]["d"]) else f"{'nan':>11}"
                    for n in names)
    tag = "CONTROL" if seed == 0 else ""
    print(f"{atoms:5d} {seed:6d} {vals}  {tag}")




print("\nmedian d_std (interval width; smaller = tighter honest intervals)")

for n in names:
    s = np.array([r["d_std"] for r in rows if r["method"] == n and np.isfinite(r["d_std"])
                  and (r["atoms"], r["seed"]) in set(bad)])
    d = np.array([r["d"] for r in rows if r["method"] == n and np.isfinite(r["d"])
                  and (r["atoms"], r["seed"]) in set(bad)])
    rec = np.mean((d > 0.5) & (d < 2.0)) if d.size else np.nan
    print(f"  {n:<11s} "
          f"median D {np.median(d):6.3f}   "
          f"median d_std {np.median(s):7.4f}   "
          f"recovered {rec:4.0%}"
    )



errs = [r for r in rows if "error" in r]
if errs:
    print("\nerrors:")
    for r in errs:
        print(f"  {r['atoms']} {r['seed']} {r['method']} -> {r['error']}")