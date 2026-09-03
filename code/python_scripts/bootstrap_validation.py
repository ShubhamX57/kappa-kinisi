"""
Evaluate a covariance constructed by resampling, and establish why it fails.

Produces the result reported in Section 5.7.4 of the dissertation. Resampling
the particles and forming the covariance of the resulting mean squared
displacement curves guarantees a positive semi definite matrix by construction,
recovers every affected case, and gives the narrowest intervals of any
treatment evaluated in this work.

Its coverage is 12.4, 17.6 and 33.8 per cent against nominal levels of 50, 68
and 95. The cause is the rank of the resampled matrix, which is bounded by the
number of particles resampled and is therefore 31 of a possible 126, so that
the construction represents no uncertainty whatever in the remaining
directions.

The treatment is included in the dissertation because it demonstrates the
necessity of the calibration check: interval width alone would have identified
it as the best of the treatments considered.

Outputs
    ~/data/bootstrap_anomalies.npy    recovery upon the fourteen affected cases
    ~/data/bootstrap_coverage.npy     coverage across healthy fits

Usage
    python bootstrap_validation.py

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


n_boot = 500
n_healthy = 1000
atoms_default = 32
length = 128



def walk(atoms, timesteps, jump_size, rng):
    """
    A three-dimensional lattice random walk.

    Each particle takes one step of the given length along one of six axis
    directions at every timestep. With a step of sqrt(6) the true diffusion
    coefficient is exactly unity, which is what makes a failed fit
    identifiable.

    """
    moves = np.zeros((6, 3))
    axis = 0
    for i in range(0, 6, 2):
        moves[i, axis] = jump_size
        moves[i + 1, axis] = -jump_size
        axis += 1
    return np.cumsum(moves[rng.choice(6, size=(atoms, timesteps))], axis=1)





def build_analyzer(seed, atoms=atoms_default):
    """
    Construct a fitted DiffusionAnalyzer for one seed.

    The trajectory is generated, wrapped in an MDAnalysis universe and passed
    to kinisi. Note that a.diff exists only after diffusion() has been called,
    so the fit is performed here rather than by the caller.

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
    Return the variances, sample counts and lag times of the diffusive
    regime.

    These are the 126 measured quantities from which the covariance is
    generated, and the objects upon which the variance treatments operate.
    """

    da = a.diff.dg['da']
    regime = a.diff.diff_regime
    return (da.data.variances[regime:],
            da.coords['n_samples'].values[regime:],
            da.coords['time interval'].values[regime:])



def cov_from_var(var, nprime):
    """
    Assemble the covariance from a variance profile.

    Applies the generating formula of the package: the entry at (i, j) is the
    variance at the shorter lag, scaled by the ratio of the two sample counts.

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

    The form gives the variance as a' n^2 / N', and the amplitude is estimated
    by weighted least squares so that the well-sampled short lags dominate.
    Fitting the amplitude rather than assuming it is what allows the treatment
    to generalise to a system whose true diffusion coefficient differs.

    """
    g = nsteps ** 2 / nprime
    w = nprime.astype(float)
    return float(np.sum(w * var * g) / np.sum(w * g * g))




def t_shrink(var, nprime, nsteps, tau_factor=1.0):
    """
    Blend each variance toward the analytical form by its sample count.

    The weight is N'/(N' + tau), so a variance supported by thousands of
    samples passes through essentially unaltered while one supported by a few
    dozen is drawn most of the way toward the model. No threshold is applied
    and nothing is discarded: the sampling decides.

    """
    ah = fit_aprime(var, nprime, nsteps)
    model = ah * nsteps ** 2 / nprime
    tau = tau_factor * float(np.median(nprime[len(nprime) // 2:]))
    w = nprime / (nprime + tau)
    return cov_from_var(w * var + (1 - w) * model, nprime)




def per_atom_msd(positions, lag_steps):
    """
    MSD per atom, averaged over time origins only.

    """
    n_atoms = positions.shape[0]
    out = np.zeros((n_atoms, len(lag_steps)))
    for k, n in enumerate(lag_steps):
        d = positions[:, n:, :] - positions[:, :-n, :]
        out[:, k] = np.mean(np.sum(d ** 2, axis=2), axis=1)
    return out





def bootstrap_cov(seed, atoms, lag_steps, n_boot=n_boot):
    """

    Covariance from resampling atoms with replacement.

    Positive semi-definite by construction, being a sum of outer products.
    Rank is limited by the number of atoms resampled, not by n_boot, so the
    matrix is typically rank deficient and needs a pseudo-inverse downstream.

    """
    rng_walk = np.random.RandomState(seed)
    pos = walk(atoms, length, np.sqrt(6), rng_walk)
    atom_msd = per_atom_msd(pos, lag_steps)

    rng_boot = np.random.default_rng(seed + 10_000_000)
    n_atoms = atom_msd.shape[0]
    reps = np.zeros((n_boot, len(lag_steps)))
    for b in range(n_boot):
        idx = rng_boot.integers(0, n_atoms, n_atoms)
        reps[b] = atom_msd[idx].mean(axis=0)
    return np.cov(reps, rowvar=False)





def realfit_with_cov(a, treated, start=2.0):
    """
    Refit a case with a treated covariance, through the genuine pipeline.

    The treated matrix is injected by monkey-patching compute_covariance_matrix
    because bayesian_regression recomputes the covariance internally, so a
    matrix supplied by any other route is silently discarded.

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
        post = a.diff.gradient.values / (2 * 3)
    finally:
        a.diff.compute_covariance_matrix = original
    return post




def cover(post, levels=(50, 68, 95)):
    """
    The proportion of fits whose posterior interval contained the truth.

    Evaluated at the nominal 50, 68 and 95 per cent levels. A treatment which
    narrows the interval without preserving these is achieving apparent
    precision through overconfidence.

    """
    out = {}
    for q in levels:
        lo, hi = np.percentile(post, [(100 - q) / 2, 100 - (100 - q) / 2])
        out[q] = bool(lo <= 1.0 <= hi)
    return out





def spectrum_report(cov):
    """
    Report the rank and spectrum of a resampled matrix.

    Included because the rank is what disqualifies the construction: it is
    bounded by the number of particles resampled, so the matrix represents no
    uncertainty at all in most of its directions.

    """
    ev = np.linalg.eigvalsh(cov)
    scale = np.abs(ev).max()
    return {"lmin": float(ev.min()),
            "lmin_rel": float(ev.min() / scale) if scale > 0 else np.nan,
            "n_neg": int((ev < 0).sum()),
            "n_neg_real": int((ev < -1e-10 * scale).sum()),
            "rank": int(np.linalg.matrix_rank(cov))}




#  part A: the 14
bad14 = [tuple(map(int, b)) for b in np.load(base / "data" / "all_bad_seeds_v2.npy")]

controls = [(16, 0), (24, 0), (32, 0), (48, 0)]

targets = bad14 + controls

ckpt_a = base / "data" / "bootstrap_anomalies.npy"



if os.path.exists(ckpt_a):
    rows_a = list(np.load(ckpt_a, allow_pickle=True))
    done_a = {(r["atoms"], r["seed"], r["method"]) for r in rows_a}
    print(f"[A] resuming - {len(rows_a)} fits done")
else:
    rows_a, done_a = [], set()


print("[A] bootstrap vs raw vs shrink on the 14 anomalies + 4 controls")
for atoms, seed in tqdm(targets):
    a = None
    for name in ("raw", "shrink", "bootstrap"):
        if (atoms, seed, name) in done_a:
            continue
        if a is None:
            a = build_analyzer(seed, atoms=atoms)
            var, nprime, nsteps = raw_pieces(a)
            lag_steps = nsteps.astype(int)

        if name == "raw":
            treated = cov_from_var(var, nprime)
        elif name == "shrink":
            treated = t_shrink(var, nprime, nsteps)
        else:
            treated = bootstrap_cov(seed, atoms, lag_steps)

        spec = spectrum_report(treated)
        try:
            post = realfit_with_cov(a, treated)
            c = cover(post)
            rows_a.append({"atoms": atoms, "seed": seed, "method": name,
                           "d": float(np.median(post)), "d_std": float(np.std(post)),
                           "c50": c[50], "c68": c[68], "c95": c[95], **spec})
        except Exception as e:
            rows_a.append({"atoms": atoms, "seed": seed, "method": name,
                           "d": np.nan, "d_std": np.nan, "c50": None,
                           "c68": None, "c95": None, **spec,
                           "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt_a, np.array(rows_a, dtype=object))




print("\n[A] recovered D (true = 1)")

names = ["raw", "shrink", "bootstrap"]

print("atoms   seed " + " ".join(f"{n:>11}" for n in names))

for atoms, seed in targets:
    g = {r["method"]: r for r in rows_a if r["atoms"] == atoms and r["seed"] == seed}
    vals = " ".join(f"{g[n]['d']:11.3f}" if n in g and np.isfinite(g[n]["d"])
                    else f"{'nan':>11}" for n in names)
    tag = "CONTROL" if seed == 0 else ""
    print(f"{atoms:5d} {seed:6d} {vals}  {tag}")


print("\n[A] bootstrap spectrum (is it really PD?)")

print("atoms   seed     lmin_rel   n_neg  real_neg   rank")

for atoms, seed in targets:
    g = [r for r in rows_a if r["atoms"] == atoms and r["seed"] == seed
         and r["method"] == "bootstrap"]
    if g:
        r = g[0]
        print(f"{atoms:5d} {seed:6d} {r['lmin_rel']:12.2e} {r['n_neg']:7d} "
              f"{r['n_neg_real']:9d} {r['rank']:6d}")










#  part B: coverage, 1000 healthy

bad32 = {295, 14460, 15770}

seeds = [s for s in range(1200) if s not in bad32][:n_healthy]

ckpt_b = base / "data" / "bootstrap_coverage.npy"

if os.path.exists(ckpt_b):
    rows_b = list(np.load(ckpt_b, allow_pickle=True))
    done_b = {(r["seed"], r["method"]) for r in rows_b}
    print(f"\n[B] resuming - {len(rows_b)} fits done")
else:
    rows_b, done_b = [], set()



print(f"\n[B] coverage on {n_healthy} healthy 32-atom seeds x (raw, shrink, bootstrap)")


for s in tqdm(seeds):
    a = None
    for name in ("raw", "shrink", "bootstrap"):
        if (s, name) in done_b:
            continue
        if a is None:
            a = build_analyzer(s, atoms=atoms_default)
            var, nprime, nsteps = raw_pieces(a)
            lag_steps = nsteps.astype(int)

        if name == "raw":
            treated = cov_from_var(var, nprime)
        elif name == "shrink":
            treated = t_shrink(var, nprime, nsteps)
        else:
            treated = bootstrap_cov(s, atoms_default, lag_steps)

        try:
            post = realfit_with_cov(a, treated)
            c = cover(post)
            rows_b.append({"seed": s, "method": name,
                           "d": float(np.median(post)), "d_std": float(np.std(post)),
                           "c50": c[50], "c68": c[68], "c95": c[95]})
        except Exception as e:
            rows_b.append({"seed": s, "method": name, "d": np.nan, "d_std": np.nan,
                           "c50": None, "c68": None, "c95": None,
                           "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt_b, np.array(rows_b, dtype=object))




print("\n[B] Results - coverage on healthy seeds (nominal 50 / 68 / 95)")

for name in ("raw", "shrink", "bootstrap"):
    g = [r for r in rows_b if r["method"] == name and np.isfinite(r.get("d", np.nan))]
    if not g:
        continue
    d = np.array([r["d"] for r in g])
    sd = np.array([r["d_std"] for r in g])
    c50 = np.mean([r["c50"] for r in g])
    c68 = np.mean([r["c68"] for r in g])
    c95 = np.mean([r["c95"] for r in g])
    cov = f"{100 * c50:.1f} / {100 * c68:.1f} / {100 * c95:.1f}"
    print(f"  {name:10s} n={len(g):4d}  median D {np.median(d):.4f}  "
          f"median d_std {np.median(sd):.4f}  coverage {cov}")



print("\n[B] summary over the 14 anomalies")
badset = set(bad14)
for name in ("raw", "shrink", "bootstrap"):
    d = np.array([r["d"] for r in rows_a if r["method"] == name
                  and (r["atoms"], r["seed"]) in badset and np.isfinite(r["d"])])
    sd = np.array([r["d_std"] for r in rows_a if r["method"] == name
                   and (r["atoms"], r["seed"]) in badset and np.isfinite(r["d_std"])])
    rec = np.mean((d > 0.5) & (d < 2.0)) if d.size else np.nan
    print(
        f"  {
            name:10s} median D {
            np.median(d):6.3f}  median d_std {
                np.median(sd):7.4f}  recovered {
                    rec:4.0%}")



errs = [r for r in rows_a + rows_b if "error" in r]
if errs:
    print(f"\nerrors: {len(errs)}")
    for r in errs[:10]:
        print(" ", r)