"""
Identify which component of the covariance is responsible for the failure.

Produces the results reported in Section 5.5 of the dissertation. Correlational
analysis having been exhausted, the responsible component is identified by
intervention: each affected matrix is decomposed, one component is replaced by
its counterpart from the analytical covariance, and the matrix is reassembled
and refitted.

Two constructions are evaluated.

    surgical_evec         replaces only the eigenvector associated with the
                          most negative eigenvalue, retaining every eigenvalue
                          and the remaining 125 directions, and
                          re-orthonormalising by QR decomposition. All fourteen
                          cases recover while the reassembled matrices remain
                          indefinite, from which it follows that indefiniteness
                          alone is not the pathology.

    negatives_only_val    replaces the eigenvalues below zero with their
                          analytical counterparts. This reduces the healthy
                          controls to zero and is therefore uninterpretable
                          it
                          is retained here because the reason for its failure
                          is instructive, and is discussed in Section 5.5.1.

Healthy controls are included in both. A construction which destroys a matrix
that was never at risk cannot be used to explain anything, and three of the
constructions evaluated in this work were disqualified on precisely that basis.

Outputs
    ~/data/surgical_swap_test.npy    one record per (case, method)

Usage
    python surgical_swap_test.py

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
    return (2 * n_steps[regime:]**2 * kappa**4) / (dim * n_samples[regime:])





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





def realfit_with_treatment(seed, treatment_fn, atoms=32, start=2.0):
    """
    Refit with a treated covariance, through the genuine pipeline.

    Injected by monkey-patching because bayesian_regression recomputes the
    covariance internally: a matrix supplied by any other route is silently
    discarded and the test measures nothing.

    """
    a = build_analyzer(seed, atoms=atoms)

    raw = build_raw_cov(a)

    treated = treatment_fn(raw)

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


true_spec = {}

for ac in [16, 24, 32, 48]:
    a0 = build_analyzer(0, atoms=ac)

    S15 = cov_from_variance_S30(a0, model_variance_S15(a0))

    true_spec[ac] = np.sort(np.linalg.eigvalsh(S15))




def surgical_evec_swap(m, U_model_lmin):
    """
    Replace the eigenvector of the most negative eigenvalue, and no more.

    Every eigenvalue is retained, including those below zero, and 125 of the 126
    directions are untouched. The basis is re-orthonormalised by QR afterwards,
    since replacing one column destroys orthogonality
    the healthy controls
    establish that the QR step is not what produces the recovery.
    """
    
    # eigh, not eig: the matrix is symmetric by construction, so the
    # eigenvalues are real and returned in ascending order

    w, V = np.linalg.eigh(m)
    if w[0] >= 0:
        return m
    u_new = U_model_lmin
    if np.dot(V[:, 0], u_new) < 0:
        u_new = -u_new
    V2 = V.copy()
    V2[:, 0] = u_new

    # replacing one column destroys orthogonality, so the basis is restored.
    # The healthy controls establish that this step is not what produces the
    # recovery: were it responsible, they would move too, and they do not.

    q, _ = np.linalg.qr(V2)

    signs = np.sign(np.sum(q * V2, axis=0))

    q = q * signs

    return (q * w) @ q.T





def negatives_only_value_swap(m, lam_true):
    """
    Replace the eigenvalues below zero with their analytical counterparts.

    Retained because the reason it fails is instructive rather than because it
    works: it reduces the healthy controls to zero, since healthy matrices carry
    negative eigenvalues of their own.
    """

    # eigh, not eig: the matrix is symmetric by construction, so the
    # eigenvalues are real and returned in ascending order
    w, V = np.linalg.eigh(m)

    neg = w < 0

    if not neg.any():
        return m
    w2 = w.copy()

    w2[neg] = lam_true[neg]

    return (V * w2) @ V.T



true_vecs_lmin = {}

for ac in [16, 24, 32, 48]:
    a0 = build_analyzer(0, atoms=ac)

    S15 = cov_from_variance_S30(a0, model_variance_S15(a0))

    # eigh, not eig: the matrix is symmetric by construction, so the
    # eigenvalues are real and returned in ascending order
    w, V = np.linalg.eigh(S15)

    true_vecs_lmin[ac] = V[:, 0]


bad = [tuple(map(int, b)) for b in np.load(base / "data" / "all_bad_seeds_v2.npy")]

controls = [(16, 0), (24, 0), (32, 0), (48, 0)]

targets = bad + controls

ckpt = base / "data" / "surgical_swap_test.npy"

if os.path.exists(ckpt):
    rows = list(np.load(ckpt, allow_pickle=True))
    done = {(r["atoms"], r["seed"], r["method"]) for r in rows}
    print(f"resuming - {len(rows)} fits done")
else:
    rows, done = [], set()


for atoms, seed in tqdm(targets):
    lam = true_spec[atoms]
    u_lmin = true_vecs_lmin[atoms]
    methods = [
        ("raw", lambda m: m),
        ("surgical_evec", lambda m, u=u_lmin: surgical_evec_swap(m, u)),
        ("negatives_only_val", lambda m, l=lam: negatives_only_value_swap(m, l)),
    ]
    for name, fn in methods:
        if (atoms, seed, name) in done:
            continue
        try:
            post = realfit_with_treatment(seed, fn, atoms=atoms)
            rows.append({"atoms": atoms, "seed": seed, "method": name,
                         "d": float(np.median(post)), "d_std": float(np.std(post))})
        except Exception as e:
            rows.append({"atoms": atoms, "seed": seed, "method": name,
                         "d": np.nan, "d_std": np.nan, "error": f"{type(e).__name__}: {e}"})
        np.save(ckpt, np.array(rows, dtype=object))


cols = ["raw", "surgical_evec", "negatives_only_val"]

print(f"\n{'atoms':>5} {'seed':>6} {'raw':>10} {'surgical_evec':>15} {'neg_only_val':>14}")

for atoms, seed in targets:
    g = {r["method"]: r for r in rows if r["atoms"] == atoms and r["seed"] == seed}

    vals = " ".join(f"{g[c]['d']:14.3f}" if c in g and np.isfinite(g[c]["d"]) else f"{'nan':>14}"
                    for c in cols)
    
    tag = "CONTROL" if seed == 0 else ""

    print(f"{atoms:5d} {seed:6d} {vals}  {tag}")


errs = [r for r in rows if "error" in r]

if errs:
    print("\nerrors:")
    for r in errs:
        print(f"  {r['atoms']} {r['seed']} {r['method']} -> {r['error']}")
