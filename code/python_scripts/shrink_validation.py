"""
Establish that the narrower intervals produced by shrinkage are honest.

Produces the results reported in Section 5.7.4 of the dissertation. A reduction
in the width of a posterior is only an improvement if the interval continues to
contain the true value as often as it claims, and a correction might otherwise
achieve apparent precision through overconfidence.

Coverage is measured across 1,000 healthy fits at 32 atoms by recording the
proportion for which the posterior interval contained unity at the nominal 50,
68 and 95 per cent levels. The untreated fit is measured under identical
conditions and serves as the reference, since it establishes the calibration of
the pipeline as it stands rather than an abstract ideal.

A sweep over the shrinkage parameter tau is included, which establishes that
the result does not depend upon its value: the sample counts span three orders
of magnitude, so the crossover between observation and model falls in the same
region for any reasonable choice.

Outputs
    ~/data/shrink_coverage.npy     one record per healthy fit and treatment
    ~/data/shrink_tau_sweep.npy    one record per (case, tau)

Cost
    Approximately three hours. Checkpointed and resumable.

Usage
    python shrink_validation.py

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





def t_shrink(var, nprime, nsteps, tau_factor=1.0):
    """
    Blend each variance toward the analytical form by its sample count.

    The weight is N'/(N' + tau): a variance from thousands of samples passes
    through essentially unaltered, one from a few dozen is drawn most of the way
    to the model. No threshold, nothing discarded
    the sampling decides.

    """
    ah = fit_aprime(var, nprime, nsteps)

    model = ah * nsteps**2 / nprime

    tau = tau_factor * float(np.median(nprime[len(nprime) // 2:]))

    w = nprime / (nprime + tau)

    return cov_from_var(w * var + (1 - w) * model, nprime)





def realfit_post(a, treated):
    """
    Refit and return the posterior samples rather than a summary.

    Coverage needs the whole posterior, since it asks how often an interval of a
    stated level contains the true value.

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
        a.diff.bayesian_regression(2.0 * sc.Unit('s'), progress=False)
        D_post = a.diff.gradient.values / (2 * 3)
    finally:
        a.diff.compute_covariance_matrix = original
    return D_post




def cover(post, levels=(50, 68, 95)):
    """
    The proportion of fits whose posterior interval contained the truth.
    
    """
    out = {}

    for q in levels:
        lo, hi = np.percentile(post, [(100 - q) / 2, 100 - (100 - q) / 2])
        out[q] = bool(lo <= 1.0 <= hi)
    return out



#  part A: shrink coverage calibration (32 atoms, 1000 healthy) 

bad32 = {295, 14460, 15770}

seeds = [s for s in range(1100) if s not in bad32][:1000]


ckpt_a = base / "data" / "shrink_coverage.npy"

if os.path.exists(ckpt_a):
    rows_a = list(np.load(ckpt_a, allow_pickle=True))
    done_a = {(r["seed"], r["method"]) for r in rows_a}
    print(f"[A] resuming - {len(rows_a)} fits done")
else:
    rows_a, done_a = [], set()


print("[A] shrink coverage calibration: 1000 healthy 32-atom seeds x {raw, shrink}")

for s in tqdm(seeds):
    a = None
    for name in ("raw", "shrink"):
        if (s, name) in done_a:
            continue
        if a is None:
            a = build_analyzer(s, atoms=32)
        var, nprime, nsteps = raw_pieces(a)
        treated = cov_from_var(var, nprime) if name == "raw" else t_shrink(var, nprime, nsteps)
        try:
            post = realfit_post(a, treated)
            c = cover(post)
            rows_a.append({"seed": s, "method": name, "d": float(np.median(post)),
                           "d_std": float(np.std(post)),
                           "c50": c[50], "c68": c[68], "c95": c[95]})
        except Exception as e:
            rows_a.append({"seed": s, "method": name, "d": np.nan, "d_std": np.nan,
                           "c50": None, "c68": None, "c95": None,
                           "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt_a, np.array(rows_a, dtype=object))

print("\n[A] RESULTS (ideal coverage: 50 / 68 / 95)")


for name in ("raw", "shrink"):
    g = [r for r in rows_a if r["method"] == name and np.isfinite(r.get("d", np.nan))]
    d = np.array([r["d"] for r in g])
    sd = np.array([r["d_std"] for r in g])
    c50 = np.mean([r["c50"] for r in g])
    c68 = np.mean([r["c68"] for r in g])
    c95 = np.mean([r["c95"] for r in g])
    print(f"  {name:7s} n={len(g)}  median D {np.median(d):.4f}  median d_std {np.median(sd):.4f}"
          f"  coverage {100 * c50:.1f} / {100 * c68:.1f} / {100 * c95:.1f}")



#  part B: tau sweep on the 14 anomalies + 4 controls 
bad = [tuple(map(int, b)) for b in np.load(base / "data" / "all_bad_seeds_v2.npy")]

controls = [(16, 0), (24, 0), (32, 0), (48, 0)]

targets = bad + controls

tau_factors = [0.1, 0.3, 1.0, 3.0, 10.0]

ckpt_b = base / "data" / "tau_sweep.npy"


if os.path.exists(ckpt_b):
    rows_b = list(np.load(ckpt_b, allow_pickle=True))
    done_b = {(r["atoms"], r["seed"], r["tau_factor"]) for r in rows_b}
    print(f"\n[B] resuming - {len(rows_b)} fits done")
else:
    rows_b, done_b = [], set()

print("\n[B] tau sweep on 14 anomalies + 4 controls")



for atoms, seed in tqdm(targets):
    a = None

    for tf in tau_factors:
        if (atoms, seed, tf) in done_b:
            continue
        if a is None:
            a = build_analyzer(seed, atoms=atoms)
        var, nprime, nsteps = raw_pieces(a)
        try:
            post = realfit_post(a, t_shrink(var, nprime, nsteps, tau_factor=tf))
            rows_b.append({"atoms": atoms, "seed": seed, "tau_factor": tf,
                           "d": float(np.median(post)), "d_std": float(np.std(post))})
        except Exception as e:
            rows_b.append({"atoms": atoms, "seed": seed, "tau_factor": tf,
                           "d": np.nan, "d_std": np.nan, "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt_b, np.array(rows_b, dtype=object))




print("\n[B] RESULTS per tau_factor (anomalies: recovery + width; controls: median D)")

badset = set(bad)

for tf in tau_factors:
    da = np.array([r["d"] for r in rows_b if r["tau_factor"] == tf
                   and (r["atoms"], r["seed"]) in badset and np.isfinite(r["d"])])
    sa = np.array([r["d_std"] for r in rows_b if r["tau_factor"] == tf
                   and (r["atoms"], r["seed"]) in badset and np.isfinite(r["d_std"])])
    dc = np.array([r["d"] for r in rows_b if r["tau_factor"] == tf
                   and r["seed"] == 0 and np.isfinite(r["d"])])
    rec = np.mean((da > 0.5) & (da < 2.0)) if da.size else np.nan
    print(f"  tau x{tf:<4}  anomaly median D {np.median(da):6.3f}  d_std {np.median(sa):7.4f}"
          f"  recovered {rec:4.0%}   control median D {np.median(dc):6.3f}")





#  part C: zero-compute reads from the existing checkpoint 
print("\n[C] zero-compute reads from variance_repair_test.npy")

vrep = list(np.load(base / "data" / "variance_repair_test.npy", allow_pickle=True))

print("  seed 295 per-method (d, d_std):")

for n in ("raw", "floor", "parametric", "shrink"):
    g = [r for r in vrep if r["atoms"] == 32 and r["seed"] == 295 and r["method"] == n]
    if g:
        print(f"    {n:11s} d {g[0]['d']:6.3f}   d_std {g[0]['d_std']:7.4f}")
print("  controls per-size, floor vs shrink (d):")


for atoms in (16, 24, 32, 48):
    gf = [r for r in vrep if r["atoms"] == atoms and r["seed"] == 0 and r["method"] == "floor"]
    gs = [r for r in vrep if r["atoms"] == atoms and r["seed"] == 0 and r["method"] == "shrink"]
    if gf and gs:
        print(f"    {atoms:2d} atoms: floor {gf[0]['d']:.3f}   shrink {gs[0]['d']:.3f}")


errs = [r for r in rows_a + rows_b if "error" in r]

if errs:
    print("\nerrors:")
    for r in errs[:10]:
        print(f"  {r}")
