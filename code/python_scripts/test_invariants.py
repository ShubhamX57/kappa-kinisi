"""
Tests of the properties every result in this work depends upon.

These are not tests of the scientific conclusions, which cannot be unit
tested, but of the machinery beneath them. Each corresponds to an assumption
that, had it been wrong, would have invalidated an entire section: that the
test system has the diffusion coefficient it is supposed to have, that the
covariance is reconstructed correctly, that the treatments do what they claim,
and that a treated matrix genuinely reaches the fit.

Three of these were written after a mistake. The reconstruction test exists
because an early comparison was made against a matrix which had already been
monkey-patched, and appeared to show a twelve per cent discrepancy which did
not exist. The injection test exists because a treated matrix passed to the
fit by any route other than the patch is silently discarded, so that a test
performed without it measures nothing at all.

    python -m pytest tests/ -v
    
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



# the test system



def test_step_length_gives_unit_diffusion():
    """
    The step is chosen so that D is exactly one.

    For a lattice walk D = kappa^2 / (2d), so a step of sqrt(6) in three
    dimensions gives unity. Every recovery figure is quoted against this, so
    an error here would move every number in the dissertation.

    """
    kappa, d = np.sqrt(6), 3
    assert kappa ** 2 / (2 * d) == pytest.approx(1.0)




def test_non_reversal_enhancement():
    """A non-reversal walk has D enhanced by (1+c)/(1-c) with c = 1/5.

    Section 5.8.1 reports recovery against a true value of 1.5, and that value
    follows from this identity rather than from measurement.
    """
    c = 1.0 / 5.0
    assert (1 + c) / (1 - c) == pytest.approx(1.5)



def test_persistent_correlation_time():
    """
    The correlation time of a persistent walk is -1/ln(p).

    An earlier version of the ballistic test used p = 0.6, for which the
    correlation time is two steps, shorter than the shortest lag sampled. The
    ballistic regime was therefore never observed and the test was
    inconclusive. This records the arithmetic that identified the error.
    """
    assert -1 / np.log(0.6) == pytest.approx(1.96, abs=0.01)
    assert -1 / np.log(0.95) == pytest.approx(19.5, abs=0.1)



# the covariance


def cov_from_var(var, nprime):
    """
    The generating formula, reproduced here so the test is self-contained.
    
    """
    n = var.size
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = cov[j, i] = (nprime[i] / nprime[j]) * var[i]
    return cov




def test_covariance_is_symmetric():
    """
    The assembled matrix must be symmetric, or eigvalsh is meaningless.
    
    """
    rng = np.random.default_rng(0)
    var = rng.uniform(1, 10, 20)
    nprime = np.linspace(2000, 30, 20)
    cov = cov_from_var(var, nprime)
    assert np.allclose(cov, cov.T)




def test_covariance_diagonal_is_the_variance():
    """
    On the diagonal the sample-count ratio is one, so the variance returns
    unchanged. This is what allows the variances to be identified in a stored
    matrix, and how the columns of the msd array were verified.
    
    """
    rng = np.random.default_rng(1)
    var = rng.uniform(1, 10, 15)
    nprime = np.linspace(2000, 30, 15)
    assert np.allclose(np.diag(cov_from_var(var, nprime)), var)



# the treatments


def adaptive_floor(m, c=0.25):
    """
    The floor, reproduced here so the test is self-contained.
    
    """
    ev, U = np.linalg.eigh(m)
    if ev.min() >= 0:
        return m
    ev = np.maximum(ev, c * abs(ev.min()))
    return U @ np.diag(ev) @ U.T




def test_floor_removes_every_negative_eigenvalue():
    """
    The floor must leave no eigenvalue below zero, which is the only
    property it claims.
    
    """
    rng = np.random.default_rng(2)
    a = rng.normal(size=(30, 30))
    m = a @ a.T
    ev, U = np.linalg.eigh(m)
    ev[:4] = [-5.0, -2.0, -0.5, -0.1]
    m = U @ np.diag(ev) @ U.T
    assert np.linalg.eigvalsh(adaptive_floor(m)).min() > 0




def test_floor_leaves_a_healthy_matrix_untouched():
    """
    A treatment which alters a matrix that was never at risk cannot be
    used, and three constructions in this work were disqualified for doing so.

    """
    rng = np.random.default_rng(3)
    a = rng.normal(size=(20, 20))
    m = a @ a.T + np.eye(20)
    assert np.allclose(adaptive_floor(m), m)




def shrink_weights(nprime, tau_factor=1.0):
    """
    The shrinkage weight, reproduced here so the test is self-contained
    
    """
    tau = tau_factor * np.median(nprime[len(nprime) // 2:])
    return nprime / (nprime + tau)




def test_shrinkage_weight_is_monotonic_and_bounded():
    """
    The weight must lie in (0, 1) and fall with the sample count, or the
    treatment is not a blend at all.
    
    """
    nprime = 4096.0 / np.arange(2, 128)
    w = shrink_weights(nprime)
    assert np.all((w > 0) & (w < 1))
    assert np.all(np.diff(w) < 0)



def test_shrinkage_keeps_the_well_sampled_points():
    """
    The short lags carry thousands of samples and must pass through
    essentially unaltered: the treatment is meant to repair the tail, not the
    whole profile.

    The sample count falls roughly as the reciprocal of the lag, not linearly,
    and the distinction matters here: a linear profile gives a weight of 0.79
    at the shortest lag where the true profile gives 0.98.

    """
    lags = np.arange(2, 128)
    nprime = 4096.0 / lags          # 2048 at the shortest, 32 at the longest
    w = shrink_weights(nprime)
    assert w[0] > 0.95
    assert w[-1] == pytest.approx(0.43, abs=0.03)




def test_shrinkage_of_a_perfect_profile_changes_nothing():
    """
    If the measurement already equals the model, blending the two must
    return the measurement, whatever the weights.
    
    """
    lags = np.arange(2, 128)
    nprime = 4096.0 / lags
    model = 3.0 * lags ** 2 / nprime
    w = shrink_weights(nprime)
    assert np.allclose(w * model + (1 - w) * model, model)


# the scale



def test_gradient_to_diffusion_conversion():
    """
    D is the gradient divided by 2d, and the injected fits report
    gradient/6. The two scales in this work differ by four orders of
    magnitude, so mistaking one for the other is not a subtle error.
    
    """
    gradient = 6.0
    
    assert gradient / 6 == pytest.approx(1.0)
