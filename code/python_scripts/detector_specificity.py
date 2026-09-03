"""
Measure the false-alarm rate of the consistency check.

Produces the results reported in Section 5.6 of the dissertation. Since no
property of the covariance identifies the affected fits, the diagnostic
compares two estimators instead: the Bayesian fit, which is weighted by the
covariance, against an ordinary least squares slope, which is not. Where the
covariance is corrupt the two diverge, and the divergence requires no knowledge
of the cause.

Sensitivity is already known from the affected cases. What this script
establishes is the cost: how often the check fires upon healthy data. One
hundred healthy fits are evaluated at each of four system sizes, which also
permits the hold-one-size-out test that every covariance-based diagnostic
fails.

The numpy slogdet warning is recorded alongside, as a candidate diagnostic
available at no computational cost. It fires upon all four hundred healthy
fits and therefore carries no information.

Outputs
    ~/data/detector_healthy.npy    one record per healthy fit: the Bayesian
                                   estimate, the least squares slope, and
                                   whether slogdet issued a warning

Usage
    python detector_specificity.py
"""



import os
import warnings
import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer
from pathlib import Path
from tqdm import tqdm
from scipy.stats import linregress

base = Path.home()

n_healthy_per_size = 100
sizes = [16, 24, 32, 48]
thresholds = [0.5, 0.6, 0.7]



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






def rawfit_with_warnflag(a, start=2.0):
    """Fit and record whether numpy issued a warning during the inversion.

    The slogdet warning suggested a diagnostic at no computational cost. It
    fires on all four hundred healthy fits and therefore carries no information.
    """
    raw = build_raw_cov(a)

    template = a.diff.compute_covariance_matrix()

    raw_var = sc.array(dims=template.dims, values=raw, unit=template.unit)

    original = a.diff.compute_covariance_matrix
    # bayesian_regression recomputes the covariance internally, so the treated
    # matrix must replace the method rather than be passed as an argument.
    # Without this the fit proceeds on the original and the test measures
    # nothing while appearing to succeed.

    a.diff.compute_covariance_matrix = lambda: raw_var

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a.diff.bayesian_regression(start * sc.Unit('s'), progress=False)
            warned = any("slogdet" in str(x.message) or x.category is RuntimeWarning for x in w)
        D_post = a.diff.gradient.values / (2 * 3)
    finally:
        a.diff.compute_covariance_matrix = original
    return D_post, warned




bad = {tuple(map(int, b)) for b in np.load(base / "data" / "all_bad_seeds_v2.npy")}


ckpt = base / "data" / "detector_healthy.npy"


if os.path.exists(ckpt):
    rows = list(np.load(ckpt, allow_pickle=True))
    done = {(r["atoms"], r["seed"]) for r in rows}
    print(f"resuming - {len(rows)} healthy fits done")
else:
    rows, done = [], set()


jobs = []

for ac in sizes:
    s = 0
    while sum(1 for j in jobs if j[0] == ac) < n_healthy_per_size:
        if (ac, s) not in bad:
            jobs.append((ac, s))
        s += 1



for atoms, seed in tqdm(jobs):
    if (atoms, seed) in done:
        continue
    try:
        a = build_analyzer(seed, atoms=atoms)
        d_ols = float(linregress(a.dt.values, a.msd.values).slope / (2 * 3))
        post, warned = rawfit_with_warnflag(a)
        rows.append({"atoms": atoms, "seed": seed, "d_ols": d_ols,
                     "d_raw": float(np.median(post)), "warned": bool(warned)})
    except Exception as e:
        rows.append({"atoms": atoms, "seed": seed, "d_ols": np.nan, "d_raw": np.nan,
                     "warned": None, "error": f"{type(e).__name__}: {e}"})
    np.save(ckpt, np.array(rows, dtype=object))



ok = [
    r for r in rows if np.isfinite(
        r.get(
            "d_raw",
            np.nan)) and np.isfinite(
                r.get(
                    "d_ols",
                    np.nan))]
ratio_h = np.array([r["d_raw"] / r["d_ols"] for r in ok])

warn_h = np.array([bool(r["warned"]) for r in ok])


anom = list(np.load(base / "data" / "anomaly_realfit_all14_v2.npy", allow_pickle=True))

araw = {(r["atoms"], r["seed"]): r for r in anom if r["method"] == "raw"}

ratio_a = np.array([r["d_raw"] / r["d_ols"] if "d_raw" in r else r["d"] / r["d_ols"]
                    for r in araw.values()])


print(f"\nhealthy fits used: {len(ok)}   anomalies: {len(ratio_a)}")


print(f"\nhealthy raw/ols ratio: median {np.median(ratio_h):.3f}, "
      f"min {ratio_h.min():.3f}, 1st pct {np.percentile(ratio_h, 1):.3f}")


print(f"anomaly raw/ols ratio: max {ratio_a.max():.3f}\n")

print(f"{'threshold':>10} {'sensitivity':>12} {'false alarms':>13} {'FPR':>7}")


for t in thresholds:
    sens = np.mean(ratio_a < t)
    fp = int(np.sum(ratio_h < t))
    print(f"{t:>10} {sens:>11.0%} {fp:>9d}/{len(ratio_h)} {fp / len(ratio_h):>7.1%}")


print(f"\nslogdet-warning detector: healthy flagged {int(warn_h.sum())}/{len(warn_h)} "
      f"({warn_h.mean():.1%})")


per = {}

for r in ok:
    per.setdefault(r["atoms"], []).append(r["d_raw"] / r["d_ols"])

print("\nper-size healthy ratio medians:")



for ac in sizes:
    v = np.array(per.get(ac, [np.nan]))
    print(f"  {ac:3d} atoms: median {np.median(v):.3f}, min {np.min(v):.3f}")



errs = [r for r in rows if "error" in r]
if errs:
    print("\nerrors:")
    for r in errs:
        print(f"  {r['atoms']} {r['seed']} -> {r['error']}")
