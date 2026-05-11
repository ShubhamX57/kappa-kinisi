# kappa-kinisi

Investigating the numerical stability of covariance matrix inversions in the `kinisi` diffusion library. Diagnosing why condition numbers are large and testing mitigation strategies.

Based on:
- [kinisi GitHub Repository](https://github.com/kinisi-dev/kinisi?utm_source=chatgpt.com)
- [kinisi Documentation](https://kinisi.readthedocs.io/en/stable/?utm_source=chatgpt.com)
- [Covariance Stabilization Paper (arXiv)](https://arxiv.org/pdf/2305.18244?utm_source=chatgpt.com)

---

# Overview

Numerical instability can arise when the covariance matrix used in diffusion analysis becomes poorly conditioned.

A covariance matrix with a large condition number ($\kappa$) is nearly singular, making matrix inversion and Cholesky decomposition unreliable. This can lead to unstable estimates of diffusion coefficients ($D^*$).

In practice, instability often appears when:

- Sampling noise dominates small eigenvalues
- Strong correlations exist between lag times
- The covariance matrix spans several orders of magnitude

---

# Diagnostics

Use NumPy to calculate the condition number of the covariance matrix.

A condition number approaching:

```math
10^{16}
