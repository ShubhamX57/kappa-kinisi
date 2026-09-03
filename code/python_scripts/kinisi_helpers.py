"""
Helper functions for the dissertation figures.

Extracted verbatim from cell 88 of Anomalous_D_Value_Inspect_full.ipynb, so
that figures generated from a script are built by the same code as the results
in the notebook. Do not edit one without the other.
"""
import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from kinisi.analyze import DiffusionAnalyzer

length = 128


def walk(atoms, timesteps, jump_size, rng):
    moves = np.zeros((6, 3))
    axis = 0
    for i in range(0, 6, 2):
        moves[i, axis] = jump_size
        moves[i + 1, axis] = -jump_size
        axis += 1
    return np.cumsum(moves[rng.choice(6, size=(atoms, timesteps))], axis=1)


def build_analyzer(seed, atoms=32):
    rng = np.random.RandomState(seed)
    steps = walk(atoms, length, np.sqrt(6), rng)
    dims = np.tile([200.0, 200.0, 200.0, 90.0, 90.0, 90.0], (steps.shape[1], 1))
    u = mda.Universe.empty(steps.shape[0], trajectory=True)
    u.add_TopologyAttr('name', [f'Atom{k}' for k in range(steps.shape[0])])
    u.add_TopologyAttr('type', ['A'] * steps.shape[0])
    u.trajectory = MemoryReader(np.transpose(steps, (1, 0, 2)),
                                dimensions=dims, delta=1.0)
    a = DiffusionAnalyzer.from_universe(
        u, time_step=1.0 * sc.Unit('s'), step_skip=1,
        distance_unit=sc.Unit('m'), specie='A',
        dt=sc.linspace(dim='time interval', start=2 * sc.Unit('s'),
                       stop=length * sc.Unit('s'), num=126),
        progress=False)
    a.diffusion(2 * sc.Unit('s'), progress=False)
    return a


def raw_pieces(a):
    da = a.diff.dg['da']
    regime = a.diff.diff_regime
    return (da.data.variances[regime:],
            da.coords['n_samples'].values[regime:],
            da.coords['time interval'].values[regime:])


def cov_from_var(var, nprime):
    n = var.size
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = (nprime[i] / nprime[j]) * var[i]
            cov[j, i] = cov[i, j]
    return cov


def build_raw_cov(a):
    var, nprime, _ = raw_pieces(a)
    return cov_from_var(var, nprime)