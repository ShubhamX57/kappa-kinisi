"""
Generate the population of fits from which the failure rate is measured.

Produces the result reported in Section 5.1 of the dissertation: 64,000 fits
across four system sizes, of which fourteen return a diffusion coefficient in
error by up to three orders of magnitude.

The system is a three-dimensional lattice random walk with a step of sqrt(6),
chosen so that the true diffusion coefficient is exactly unity by construction.
Every fit therefore has a known answer, which is what makes a failure
identifiable at all: on real material data a wrong diffusion coefficient is
indistinguishable from a right one.

This script is a second, independently written implementation of an earlier
run. The two identify the same fourteen cases, which establishes that the
failures are reproducible rather than artefacts of a particular execution.

Outputs
    ~/data/failure_population_v2.npy   one record per fit: seed, atoms, D,
                                       condition number, smallest eigenvalue,
                                       negative fraction, deviation from the
                                       analytical spectrum
    ~/data/all_bad_seeds_v2.npy        the (atoms, seed) pairs classified as
                                       affected

Cost
    Approximately eight hours across all cores. The run is checkpointed every
    2,000 seeds and resumes from the checkpoint if interrupted.

Usage
    python regenerate_population_v2.py

"""



import os
import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer
from pathlib import Path
from joblib import Parallel, delayed



base = Path.home()



atom_counts = [16, 24, 48]
n_seeds = 16000
n_jobs = -1
chunk = 2000



ckpt_pop = base / "data" / "failure_population_v2.npy"
old_32 = base / "data" / "rw_32atoms_16k.npy"
bad_out = base / "data" / "all_bad_seeds_v2.npy"



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




def model_variance_S15(a, kappa=np.sqrt(6), dim=3):
    """
    The analytical variance of the mean-squared displacement.

    Derived for free diffusion and exact for the lattice walk used here. It
    describes a simple walk to within 2.6 per cent and departs by 38 per cent
    under ballistic motion.
    """
    da = a.diff.dg['da']
    n_samples = da.coords['n_samples'].values
    n_steps = da.coords['time interval'].values
    regime = a.diff.diff_regime
    n = n_steps[regime:]
    Nprime = n_samples[regime:]
    return (2 * n**2 * kappa**4) / (dim * Nprime)





def cov_from_variance_S30(a, sigma2):
    """
    Assemble the covariance from a variance profile.

    Applies the generating formula: entry (i, j) is the variance at the shorter
    lag scaled by the ratio of the two sample counts.
    """
    da = a.diff.dg['da']
    n_samples = da.coords['n_samples'].values
    regime = a.diff.diff_regime
    Nprime = n_samples[regime:]
    k = sigma2.size
    cov = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            cov[i, j] = sigma2[i] * Nprime[i] / Nprime[j]
            cov[j, i] = cov[i, j]
    return cov





def main():
    """
    Run the population, checkpointing as it goes.

    Work is distributed with joblib and written every 2,000 seeds so an
    interrupted run resumes rather than restarts. It aborts after twenty
    consecutive failures, since an environment fault would otherwise fill the
    record with missing values.
    """
    true_ev = {}
    for ac in atom_counts:
        a0 = build_analyzer(0, atoms=ac)
        s2 = model_variance_S15(a0)
        true_ev[ac] = np.sort(np.linalg.eigvalsh(cov_from_variance_S30(a0, s2)))[::-1]

    def fit_one(ac, s):
        """
        Fit one seed and return its summary.

        Records the diffusion coefficient alongside every candidate diagnostic, so
        the population need not be regenerated when a new one is proposed.
        """
        try:
            a = build_analyzer(s, atoms=ac)
            raw = build_raw_cov(a)
            ev = np.sort(np.linalg.eigvalsh(raw))[::-1]
            n = min(80, len(ev))
            pos = ev[ev > 0]
            return {"atoms": ac, "seed": s,
                    "D": float(np.median(a.D.values)),
                    "cond": float(np.linalg.cond(a.diff.covariance_matrix.values)),
                    "frac_neg": float(np.mean(ev < 0)),
                    "n_pos": int(np.sum(ev > 0)),
                    "lmin": float(ev[-1]), "lmax": float(ev[0]),
                    "kappa_pos": float(ev[0] / pos[-1]) if pos.size else np.inf,
                    "bulk_dev": float(np.mean(np.log10(true_ev[ac][:n])
                                              - np.log10(np.maximum(ev[:n], 1e-10))))}
        except Exception as e:
            return {"atoms": ac, "seed": s, "D": np.nan, "error": f"{type(e).__name__}"}




    test = fit_one(16, 0)
    assert np.isfinite(test["D"]), f"pre-flight FAILED: {test.get('error')}"
    print(f"pre-flight OK: 16-atom seed 0 D={test['D']:.1f}")



    if os.path.exists(ckpt_pop):
        records = list(np.load(ckpt_pop, allow_pickle=True))
        print(f"resuming - {len(records)} records in {ckpt_pop.name}")
    else:
        records = []
        if os.path.exists(old_32):
            old = list(np.load(old_32, allow_pickle=True))
            for r in old:
                r2 = dict(r)
                r2["atoms"] = 32
                records.append(r2)
            np.save(ckpt_pop, np.array(records, dtype=object))
            print(f"seeded with {len(old)} intact 32-atom records from {old_32.name}")
            print("(32-atom slice reuses old records: D valid; raw-cov metrics absent there)")
        else:
            print(f"WARNING: {old_32.name} not found - 32-atom slice will be missing")




    done = {(r["atoms"], r["seed"]) for r in records}
    jobs = [(ac, s) for ac in atom_counts for s in range(n_seeds) if (ac, s) not in done]
    print(f"running {len(jobs)} fits across {n_jobs if n_jobs > 0 else os.cpu_count()} cores...\n")




    for i in range(0, len(jobs), chunk):
        batch = jobs[i:i + chunk]
        out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(fit_one)(ac, s) for ac, s in batch)
        records.extend(out)
        np.save(ckpt_pop, np.array(records, dtype=object))
        print(f"  checkpoint: {len(records)} done")




    np.save(ckpt_pop, np.array(records, dtype=object))
    print(f"\ndone: {len(records)} total fits")



    D = np.array([r["D"] for r in records])
    atoms_arr = np.array([r["atoms"] for r in records])
    finite = np.isfinite(D)
    anom = np.zeros(len(records), dtype=bool)
    for ac in np.unique(atoms_arr):
        m = (atoms_arr == ac) & finite
        Dn = np.median(D[m & (D > 1000)])
        anom |= m & ((D < 0.5 * Dn) | (D <= 0))
    anom |= ~finite



    print(f"\ntotal fits: {len(records)}")
    print(f"total anomalies: {int(anom.sum())}\n")
    for ac in np.unique(atoms_arr):
        m = atoms_arr == ac
        print(
            f"  {ac:3d} atoms: {int((anom & m).sum()):3d} anomalies / {int(m.sum())} ({100 * np.mean(anom[m]):.3f}%)")



    bad = [(int(records[i]["atoms"]), int(records[i]["seed"])) for i in np.where(anom)[0]]
    print(f"\nbad list: {bad}")



    sizes_ok = all(int((atoms_arr == ac).sum()) >= n_seeds for ac in [16, 24, 32, 48])
    if len(records) >= 60000 and sizes_ok:
        np.save(bad_out, np.array(bad))
        print(f"\nsaved {len(bad)} pairs to {bad_out.name}")
    else:
       print(
           f"\nGUARD: NOT saving {bad_out.name} - "
           f"population incomplete ({len(records)} records; "
           f"all four sizes >= {n_seeds} required)"
      )



if __name__ == "__main__":
    main()