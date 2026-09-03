"""
Test whether the correction generalises to a system with a different answer.

Produces the results reported in Section 5.8.1 of the dissertation. A
non-reversal walk, in which a step may not immediately retrace its predecessor,
remains diffusive at long times but possesses a different diffusion
coefficient: for a cubic lattice the mean cosine between consecutive steps is
1/5, giving a correlation factor of (1+c)/(1-c) = 1.5.

The significance of the test lies in the fact that the amplitude of the
analytical variance form is estimated from the data, so the correction is not
informed that the correct answer has changed. It recovers 1.477 against a true
value of 1.5, upon a system possessing a different correlation structure from
any upon which it was developed.

Part A generates the population and identifies the affected cases
Part B
applies the corrections to them, against the revised true value.

Outputs
    ~/data/nonreversal_population.npy    one record per fit
    ~/data/nonreversal_repair.npy        recovery under each treatment

Cost
    Approximately thirty minutes on eight cores. Checkpointed every 2,000 fits.
    Set the thread limits before running, or the workers oversubscribe:

        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python nonreversal_recovery_study.py

Usage
    See above. To stop, kill both the parent process and the loky workers.

"""



import os
import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm


base = Path.home()
length = 128
n_seeds = 4000
atom_counts = [16, 24, 32, 48]
n_jobs = 8
chunk = 2000




def walk_nonreversal(atoms, timesteps, jump_size, rng):
    """
    Random walk that cannot step directly back where it came from.

    Move indices are paired so that the reverse of move i is i ^ 1. At each
    step we draw one of the five permitted directions by drawing 0..4 and
    skipping over the forbidden index.

    """


    moves = np.zeros((6, 3))
    axis = 0

    for i in range(0, 6, 2):
        moves[i, axis] = jump_size
        moves[i + 1, axis] = -jump_size
        axis += 1


    choices = np.zeros((atoms, timesteps), dtype=int)
    choices[:, 0] = rng.choice(6, size=atoms)

    for t in range(1, timesteps):
        forbidden = choices[:, t - 1] ^ 1
        r = rng.choice(5, size=atoms)
        choices[:, t] = r + (r >= forbidden)

    return np.cumsum(moves[choices], axis=1)





def build_nr(seed, atoms=32):
    """
    A fitted analyser for one non-reversal walk.

    Identical to the simple-walk case apart from the generator, so that any
    difference in outcome is attributable to the step correlation rather than to
    the analysis.

    """
    rng = np.random.RandomState(seed)

    steps = walk_nonreversal(atoms, length, np.sqrt(6), rng)

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
        fl = c * (-v[0])
        v = np.where(v < fl, fl, v)
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




def fit_one(ac, s):
    """
    One population fit. Must be module-level so joblib can pickle it.
    
    """
    try:
        a = build_nr(s, atoms=ac)
        return {"atoms": ac, "seed": s,
                "D": float(np.median(a.D.values)),
                "D_std": float(np.std(a.D.values))}
    except Exception as e:
        return {"atoms": ac, "seed": s, "D": np.nan, "D_std": np.nan,
                "error": f"{type(e).__name__}: {e}"}




if __name__ == "__main__":

    #  part A: population
    # The true D for a non-reversal walk is 1.5x the simple-walk value, from the
    # correlation factor (1+c)/(1-c) with c = 1/5. Verified numerically at 1.483.
    # Anomalies are judged against the per-size median, as before, so the absolute
    # scale does not matter here.


    ckpt_a = base / "data" / "nonreversal_population.npy"

    if os.path.exists(ckpt_a):
        rows_a = list(np.load(ckpt_a, allow_pickle=True))
        done_a = {(r["atoms"], r["seed"]) for r in rows_a}
        print(f"[A] resuming - {len(rows_a)} fits done")
    else:
        rows_a, done_a = [], set()


    jobs = [(ac, s) for ac in atom_counts for s in range(n_seeds)
            if (ac, s) not in done_a]
    
    print(f"[A] {len(jobs)} fits to run on {n_jobs} cores")


    for i in range(0, len(jobs), chunk):
        batch = jobs[i:i + chunk]
        out = Parallel(n_jobs=n_jobs, verbose=5, batch_size=8)(
            delayed(fit_one)(ac, s) for ac, s in batch)
        rows_a.extend(out)
        np.save(ckpt_a, np.array(rows_a, dtype=object))
        print(f"[A] checkpointed {len(rows_a)} fits")


    print("\n[A] anomalies by system size")

    bad_nr = []

    for ac in atom_counts:
        g = [r for r in rows_a if r["atoms"] == ac]
        d = np.array([r["D"] for r in g])
        fin = np.isfinite(d)
        typical = np.median(d[fin & (d > 0)])
        flag = (~fin) | (d <= 0) | (d < 0.5 * typical) | (d > 2 * typical)

        for i in np.where(flag)[0]:
            bad_nr.append((ac, int(g[i]["seed"])))

        print(f"  {ac:2d} atoms: median D {typical:8.1f}   anomalies {int(flag.sum()):3d}"
              f"   ({100 * flag.mean():.3f}%)")

    np.save(base / "data" / "nonreversal_bad_seeds.npy", np.array(bad_nr))

    print(f"\n[A] total anomalies: {len(bad_nr)} of {len(rows_a)}")

    #  part B: the repairs
    if not bad_nr:
        print("\n[B] no anomalies found - nothing to repair. This is itself a result:")
        print("    report the rate and stop here.")
        raise SystemExit


    controls = [(ac, 0) for ac in atom_counts]

    targets = bad_nr + controls

    ckpt_b = base / "data" / "nonreversal_repair.npy"

    if os.path.exists(ckpt_b):
        rows_b = list(np.load(ckpt_b, allow_pickle=True))
        done_b = {(r["atoms"], r["seed"], r["method"]) for r in rows_b}
        print(f"\n[B] resuming - {len(rows_b)} fits done")
    else:
        rows_b, done_b = [], set()


    print(f"\n[B] repairs on {len(bad_nr)} anomalies + {len(controls)} controls")

    for atoms, seed in tqdm(targets):
        todo = [m for m in ("raw", "floor", "shrink")
                if (atoms, seed, m) not in done_b]
        if not todo:
            continue

        a = build_nr(seed, atoms=atoms)
        var, nprime, nsteps = raw_pieces(a)
        built = {"raw": cov_from_var(var, nprime),
                 "floor": adaptive_floor(cov_from_var(var, nprime)),
                 "shrink": cov_from_var(shrunk_var(var, nprime, nsteps), nprime)}

        for name in todo:
            cov = built[name]
            ev = np.linalg.eigvalsh(cov)
            try:
                post = realfit_with_cov(a, cov)
                rows_b.append({"atoms": atoms, "seed": seed, "method": name,
                               "d": float(np.median(post)), "d_std": float(np.std(post)),
                               "n_neg": int((ev < 0).sum())})
            except Exception as e:
                rows_b.append({"atoms": atoms, "seed": seed, "method": name,
                               "d": np.nan, "d_std": np.nan, "n_neg": int((ev < 0).sum()),
                               "error": f"{type(e).__name__}: {e}"})
            np.save(ckpt_b, np.array(rows_b, dtype=object))


    # The injected fits read gradient/6, normalised so a SIMPLE walk gives 1.
    # A non-reversal walk should therefore give 1.5 when recovered correctly.

    target = 1.5

    methods = ["raw", "floor", "shrink"]


    print(f"\n[B] recovered D  (correct answer is {target}, not 1)")

    print(f"{'atoms':>5} {'seed':>6} " + " ".join(f"{m:>10}" for m in methods))

    for atoms, seed in targets:
        g = {r["method"]: r for r in rows_b if r["atoms"] == atoms and r["seed"] == seed}
        vals = " ".join(f"{g[m]['d']:10.3f}" if m in g and np.isfinite(g[m]["d"])
                        else f"{'nan':>10}" for m in methods)
        print(f"{atoms:5d} {seed:6d} {vals}  {'CONTROL' if seed == 0 else ''}")


    print("\n[B] summary over the anomalies")

    bad_set = set(bad_nr)

    for m in methods:
        d = np.array([r["d"] for r in rows_b if r["method"] == m
                      and (r["atoms"], r["seed"]) in bad_set and np.isfinite(r["d"])])
        sd = np.array([r["d_std"] for r in rows_b if r["method"] == m
                       and (r["atoms"], r["seed"]) in bad_set and np.isfinite(r["d_std"])])
        rec = np.mean((d > 0.5 * target) & (d < 2 * target)) if d.size else np.nan
        print(f"  {m:8s} median D {np.median(d):6.3f}  (target {target})"
              f"   median d_std {np.median(sd):7.4f}   recovered {rec:4.0%}")


    print(f"\n[B] controls (should sit near {target})")

    for m in methods:
        d = np.array([r["d"] for r in rows_b if r["method"] == m
                      and r["seed"] == 0 and np.isfinite(r["d"])])
        print(f"  {m:8s} " + "  ".join(f"{x:.3f}" for x in d))

    errs = [r for r in rows_a + rows_b if "error" in r]

    if errs:
        print(f"\nerrors: {len(errs)}")
        for r in errs[:5]:
            print(" ", r)
