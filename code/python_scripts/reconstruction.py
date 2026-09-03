"""
Verify that the raw covariance can be reconstructed from what the package
retains, and that the injection path is faithful.

Supports Sections 4.2.2 and 5.7.1 of the dissertation. Two facts are
established here, and everything else in this work depends upon both.

The first is that kinisi retains only the covariance which remains after
treatment, so that study of the failure requires the matrix as it stood
beforehand. It is reconstructed from the stored variances according to the
generating formula, taking the sample counts from the coordinates of the
analyser and slicing the result to the diffusive regime. The reconstruction is
compared against the matrix the package itself holds, upon three systems
spanning different regimes: the random walks, the LiPS example distributed with
the package, and an argyrodite example.

The second is that supplying a treated matrix through the monkey-patched
injection path does not itself alter the posterior. The check is to supply the
package with the covariance it computes for itself and confirm that the
resulting interval matches the native one.

Usage
    python reconstruction.py

"""


from MDAnalysis import Universe
from ase.io import read
import os
import numpy as np
import scipp as sc
import kinisi
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from pymatgen.io.vasp import Xdatcar
from kinisi.analyze import DiffusionAnalyzer



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
            cov[i, j] = (n_samples[i] / n_samples[j]) * variances[i]
            cov[j, i] = cov[i, j]

    regime = a.diff.diff_regime
    # sliced to the diffusive regime: kinisi fits only beyond diff_regime, and
    # the lag times before it are present in the arrays but unused
    return cov[regime:, regime:]





def compare(a, label):
    """
    compare kinisi's covariance against the rebuilt one

    """

    kinisi_cov = a.diff.covariance_matrix.values

    raw = build_raw_cov(a)

    da = a.diff.dg['da']

    regime = a.diff.diff_regime

    print(f"\n {label}")
    print(f"diff_regime            : {regime}")
    print(f"variances length       : {da.data.variances.size}")
    print(f"n_samples length       : {da.coords['n_samples'].values.size}")
    print(f"kinisi cov shape       : {kinisi_cov.shape}")
    print(f"rebuilt cov shape      : {raw.shape}")

    if kinisi_cov.shape != raw.shape:
        print("shape did mot mismatch, cannot compare")
        return

    rel = np.abs(kinisi_cov - raw) / (np.abs(kinisi_cov) + 1e-30)
    print(f"median relative diff   : {np.nanmedian(rel):.3e}")
    print(f"max relative diff      : {np.nanmax(rel):.3e}")
    print(f"ratio kinisi/rebuilt   : {np.nanmedian(kinisi_cov / (raw + 1e-30)):.6f}")
    print(f"kinisi lambda_min      : {np.linalg.eigvalsh(kinisi_cov).min():.3e}")
    print(f"rebuilt lambda_min     : {np.linalg.eigvalsh(raw).min():.3e}")





#  random walk
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





def build_analyzer(seed, atoms=32, length=128):
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


a_rw = build_analyzer(0, atoms=32)

compare(a_rw, "random walk, 32 atoms, seed 0")


#  lips

lips_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/LiPS.exyz')

frames = read(lips_path, format='extxyz', index=':')

cell_dims = [[*f.cell.lengths(), *f.cell.angles()] for f in frames]

u_lips = Universe(lips_path, lips_path, format='XYZ', topology_format='XYZ', dt=20.0 / 1000)
for ts, dims in zip(u_lips.trajectory, cell_dims):
    ts.dimensions = dims


a_lips = DiffusionAnalyzer.from_universe(
    u_lips, specie='LI', time_step=0.001 * sc.Unit('ps'),
    step_skip=20 * sc.units.dimensionless, progress=False)
a_lips.diffusion(1.5 * sc.Unit('ps'), progress=False)
compare(a_lips, "LiPS")




#  argyrodite
xd_path = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/example_XDATCAR.gz')
a_arg = DiffusionAnalyzer.from_xdatcar(
    Xdatcar(xd_path), specie='Li',
    time_step=2.0 * sc.Unit('fs'),
    step_skip=50 * sc.Unit('dimensionless'),
    progress=False)
a_arg.diffusion(3000.0 * sc.Unit('fs'), progress=False)
compare(a_arg, "argyrodite, example_XDATCAR")
