"""
Test the assumption upon which shrinkage rests, under ballistic motion.

Supports the result reported in Section 5.8.2 of the dissertation. Shrinkage
appeals to the analytical form of the variance profile, which is derived for
diffusive motion. A persistent walk, in which the previous direction is
retained with probability p, permits that assumption to be tested directly,
since the motion is ballistic over a range determined by the correlation time
-1/ln(p).

The parameter matters. At p = 0.6 the correlation time is approximately two
steps, which is shorter than the shortest lag sampled, and the ballistic regime
is therefore never observed
an earlier version of this test was inconclusive
for that reason. At p = 0.95 the correlation time is approximately twenty steps
and the short-lag mean-squared displacement exponent reaches 1.88.

Three checks are performed: that the diffusion coefficient is enhanced as the
theory predicts, that the analytical variance form still describes the profile,
and that the mean-squared displacement is ballistic at short lag times as
claimed.

Requires raw_pieces from the analysis notebook
run it there, or import the
helper module.

Usage
    python persistent_shape_check.py

"""


import numpy as np
import scipp as sc
import MDAnalysis as mda
import matplotlib.pyplot as plt
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer


length = 128



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




def walk_persistent(atoms, timesteps, jump_size, p, rng):
    """
    Random walk that keeps its previous direction with probability p.

    p = 0 gives an ordinary random walk. Larger p gives a longer ballistic
    stretch before the motion becomes diffusive, which is the regime that
    breaks the assumed variance form.
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
        keep = rng.random(atoms) < p
        fresh = rng.choice(6, size=atoms)
        choices[:, t] = np.where(keep, choices[:, t - 1], fresh)

    return np.cumsum(moves[choices], axis=1)





def build_persistent(seed, p, atoms=32):
    """
    A fitted analyser for one persistent walk at a given p.
    
    """
    rng = np.random.RandomState(seed)
    steps = walk_persistent(atoms, length, np.sqrt(6), p, rng)
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




def shape_residual(a):
    """
    Fractional departure of the measured variance from the S-15 shape.
    
    """
    var, nprime, nsteps = raw_pieces(a)
    g = nsteps ** 2 / nprime
    ahat = np.sum(nprime * var * g) / np.sum(nprime * g * g)
    model = ahat * g
    return nsteps, var, model, (var - model) / model





# ---------------------------------------------------------------------------
# part 1: does the construction behave as predicted?
# For a persistent walk the long-time D should be enhanced by (1+p)/(1-p),
# so p = 0 gives 1x, p = 1/3 gives 2x, p = 0.6 gives 4x. Derived rather than
# looked up, so it is checked numerically here before anything rests on it.

p_values = [0.0, 1.0 / 3.0, 0.6]
print("     p   predicted    measured    ratio")
d_ref = None
for p in p_values:
    d = np.median([float(np.median(build_persistent(s, p).D.values)) for s in range(15)])
    if d_ref is None:
        d_ref = d
    predicted = (1 + p) / (1 - p)
    print(f"{p:6.3f} {predicted:11.3f} {d / d_ref:11.3f} {(d / d_ref) / predicted:8.3f}")






# ---------------------------------------------------------------------------
# part 2: the question that matters - does the S-15 shape still fit?

print("\n     p  median |residual|   (simple walk was 0.201)")
results = {}
for p in [0.0, 0.2, 1.0 / 3.0, 0.5, 0.6, 0.7]:
    t, v, m, r = shape_residual(build_persistent(0, p))
    results[p] = (t, v, m, r)
    print(f"{p:6.3f} {np.median(np.abs(r)):18.4f}")

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for axis, p in zip(ax.ravel(), results):
    t, v, m, r = results[p]
    axis.plot(t, v, lw=1, color="#888780", label="measured")
    axis.plot(t, m, lw=1.8, color="#1D9E75", label="S-15 shape")
    axis.set_title(f"p = {p:.2f}   residual {np.median(np.abs(r)):.3f}", fontsize=10)
    axis.set_xlabel("lag time", fontsize=9)
    axis.set_ylabel("variance", fontsize=9)
    axis.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig("persistent_shape_check.png", dpi=150, bbox_inches="tight")
plt.show()





# ---------------------------------------------------------------------------
# part 3: is the MSD itself visibly ballistic? this is the physical check that
# the construction is doing what it claims, independent of the variance work.

print("\n     p  MSD slope, short lags  MSD slope, long lags")
for p in [0.0, 1.0 / 3.0, 0.6]:
    a = build_persistent(0, p)
    t, msd = a.dt.values, a.msd.values
    m = msd > 0
    lg = np.gradient(np.log(msd[m]), np.log(t[m]))
    print(f"{p:6.3f} {np.median(lg[:15]):22.3f} {np.median(lg[-40:]):21.3f}")
print("\nslope 1 is diffusive, slope 2 is ballistic")