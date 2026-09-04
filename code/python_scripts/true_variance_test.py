"""
Measure the true sampling variance directly, without appeal to any model.

Produces the results reported in Section 5.8.2 of the dissertation. The
variance which the covariance diagonal is meant to represent is the sampling
variance of the mean-squared displacement estimate, and it may be measured by
brute force: generate many independent realisations and take the variance of
the resulting curves across them.

The measurement requires no model and no known diffusion coefficient, which
makes it applicable to systems for which neither is available. It establishes
that the analytical form describes the true variance of a simple walk to within
2.6 per cent, and of a persistent walk at p = 0.95 to within 38.2 per cent.

One comparison in this script should not be relied upon. The single-walk
estimator used for the toward-or-away test does not match the estimator
employed by the package, and gives incoherent results even upon the simple
walk, where shrinkage is known to work. The shape comparison is valid
the
toward-or-away verdict is not, and is not reported in the dissertation.

Usage
    python true_variance_test.py
"""


import numpy as np
import matplotlib.pyplot as plt
from kinisi_helpers import build_analyzer, raw_pieces



length = 128
atoms = 32
n_real = 2000
lags = np.arange(2, 128)


def moves_table(jump_size):
    """
    The six unit steps of a cubic lattice, scaled to the jump length.
    """
    m = np.zeros((6, 3))
    axis = 0
    for i in range(0, 6, 2):
        m[i, axis] = jump_size
        m[i + 1, axis] = -jump_size
        axis += 1
    return m




def walk_simple(atoms, timesteps, jump_size, rng):
    """
    An uncorrelated lattice random walk, for reference.
    """
    m = moves_table(jump_size)
    return np.cumsum(m[rng.choice(6, size=(atoms, timesteps))], axis=1)




def walk_persistent(atoms, timesteps, jump_size, p, rng):
    """
    Keeps its previous direction with probability p.

    Correlation time is -1/ln(p), so p = 0.95 gives about 20 steps and the
    ballistic stretch actually covers part of the lag range. At p = 0.6 it is
    only 2 steps, which is why the earlier attempt saw nothing.

    """
    m = moves_table(jump_size)
    choices = np.zeros((atoms, timesteps), dtype=int)
    choices[:, 0] = rng.choice(6, size=atoms)
    for t in range(1, timesteps):
        keep = rng.random(atoms) < p
        choices[:, t] = np.where(keep, choices[:, t - 1], rng.choice(6, size=atoms))
    return np.cumsum(m[choices], axis=1)




def msd_curve(positions, lags):
    """
    MSD averaged over atoms and over all time origins, as kinisi does.
    
    """
    out = np.zeros(len(lags))
    for k, n in enumerate(lags):
        d = positions[:, n:, :] - positions[:, :-n, :]
        out[k] = np.mean(np.sum(d ** 2, axis=2))
    return out



def n_samples(lags, atoms, timesteps):
    """
    Independent samples behind each lag, as kinisi counts them.
    
    """
    return atoms * (timesteps - lags)



def true_variance(walk_fn, n_real, seed0=0):
    """
    The sampling variance of the MSD estimate, measured rather than modelled.

    Generates many independent realisations and takes the variance across them.
    This is the quantity the covariance diagonal is supposed to represent, and
    it needs no assumption about the form of the motion.
    """
    curves = np.zeros((n_real, len(lags)))
    for i in range(n_real):
        rng = np.random.RandomState(seed0 + i)
        curves[i] = msd_curve(walk_fn(rng), lags)
    return curves.mean(axis=0), curves.var(axis=0, ddof=1), curves



def fit_s15(var, nprime):
    """
    Fit the S-15 amplitude, weighting by sample count as shrink does.
    
    """
    g = lags ** 2 / nprime
    ahat = np.sum(nprime * var * g) / np.sum(nprime * g * g)
    return ahat * g, ahat




def shrink(var, nprime, tau_factor=1.0):
    """
    Blend a variance profile toward the fitted analytical form.
    
    """
    model, _ = fit_s15(var, nprime)
    tau = tau_factor * np.median(nprime[len(nprime) // 2:])
    w = nprime / (nprime + tau)
    return w * var + (1 - w) * model




# The sample counts must be those the package uses, since the analytical form
# is fitted against them. They are taken from the analyser rather than
# recomputed: an earlier version used a formula of its own which diverged at the
# longest lags and made the fitted amplitude meaningless.

_ref = build_analyzer(0, atoms=atoms)
_, nprime, _ = raw_pieces(_ref)



systems = {
    "simple walk": lambda rng: walk_simple(atoms, length, np.sqrt(6), rng),
    "persistent p=0.95": lambda rng: walk_persistent(atoms, length, np.sqrt(6), 0.95, rng),
}



print(f"measuring the true variance from {n_real} independent realisations each\n")
results = {}
for name, fn in systems.items():
    msd_mean, var_true, curves = true_variance(fn, n_real)
    model, ahat = fit_s15(var_true, nprime)
    shape_err = np.median(np.abs(var_true - model) / model)

    # how ballistic is it? slope 1 is diffusive, 2 is ballistic
    lg = np.gradient(np.log(msd_mean), np.log(lags.astype(float)))
    results[name] = dict(var_true=var_true, model=model, curves=curves,
                         msd_mean=msd_mean, slope_short=np.median(lg[:12]),
                         slope_long=np.median(lg[-40:]), shape_err=shape_err)


    print(f"{name}")
    print(f"  MSD slope, short lags : {np.median(lg[:12]):.3f}   (1 diffusive, 2 ballistic)")
    print(f"  MSD slope, long lags  : {np.median(lg[-40:]):.3f}")
    print(f"  S-15 fits the TRUE variance to : {shape_err:.4f}")
    print()


print("does shrink move a single noisy estimate toward the truth or away from it?\n")
print("                system   raw error  shrunk error      verdict")

for name, fn in systems.items():
    var_true = results[name]["var_true"]

    raw_err, shr_err = [], []
    for s in range(200, 240):
        rng = np.random.RandomState(s)
        pos = fn(rng)

        # the variance one walk would estimate, from the scatter across atoms
        per_atom = np.zeros((atoms, len(lags)))
        for k, n in enumerate(lags):
            d = pos[:, n:, :] - pos[:, :-n, :]
            per_atom[:, k] = np.mean(np.sum(d ** 2, axis=2), axis=1)
        var_est = per_atom.var(axis=0, ddof=1) / atoms

        var_shr = shrink(var_est, nprime)
        raw_err.append(np.median(np.abs(var_est - var_true) / var_true))
        shr_err.append(np.median(np.abs(var_shr - var_true) / var_true))

    r, s_ = np.median(raw_err), np.median(shr_err)
    verdict = "closer" if s_ < r else "FURTHER"
    print(f"{name:22} {r:11.4f} {s_:13.4f} {verdict:>12}")
    results[name]["raw_err"], results[name]["shr_err"] = r, s_




fig, ax = plt.subplots(2, 2, figsize=(13, 9))

for col, (name, res) in enumerate(results.items()):
    a = ax[0, col]
    a.plot(lags, res["msd_mean"], lw=1.6, color="#378ADD")
    a.plot(lags, res["msd_mean"][0] * (lags / lags[0]), "k--", lw=1, label="slope 1")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_title(f"{name}\nshort-lag slope {res['slope_short']:.2f}", fontsize=10)
    a.set_xlabel("lag")
    a.set_ylabel("MSD")
    a.legend(frameon=False, fontsize=8)


    a = ax[1, col]
    a.plot(lags, res["var_true"], lw=1.4, color="#888780", label="true variance")
    a.plot(lags, res["model"], lw=2, color="#1D9E75", label="S-15 shape")
    a.set_title(f"S-15 fits truth to {res['shape_err']:.3f}", fontsize=10)
    a.set_xlabel("lag")
    a.set_ylabel("variance")
    a.legend(frameon=False, fontsize=8)


plt.tight_layout()
plt.savefig("true_variance_test.png", dpi=150, bbox_inches="tight")
plt.show()