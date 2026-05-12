# Project log : kappa-kinisi
---
## 12/05/2026 (Tue) — Covariance deep-dive

**Did**: Worked through covariance matrix material — eigenvalues,
condition number, and Mahalanobis distance.

**Learned**: [Pick one — e.g., "An ill-conditioned matrix means the
data is squashed in one direction, and that direction is where
inversion blows up errors."]

**Next session**: Re-read Section 3 of the paper with this grounding,
then start T1.2 — annotate Eq. 5.

---
## 11/05/2025 (Mon) -   Completed core papers 

**Did**: i) Tabeart et al. (2022) — "Improving the condition number of estimated covariance matrices" — this is the actual reference for the reconditioning method kinisi uses (paper ref 33). I am going to be testing alternatives to this method, so i must know what it does in detail.
ii) Usler, Kemp, Bonkowski & De Souza (2023) — "A general expression for the statistical error in a diffusion coefficient" — paper ref 21 in McCluskey. Companion paper that frames why OLS uncertainty is wrong for MSD data.

**Next session**: Going with reading more core papers.


---
## 08/05/2026 (Fri) — Covariance video

**Did**: Watched a video on covariance matrices to start
building the grounding needed for Section 3 of the paper.

**Next session**: Continue covariance deep dive — eigenvalues,
condition number, and Mahalanobis distance — then re-read Section 3
of the paper with that grounding.

---

## 07/05/2026 (Thu) — T1.1 done + planning covariance deep-dive

**Did**: Read McCluskey/Coles/Morgan paper end-to-end (T1.1).
Identified the covariance matrix as the conceptual bottleneck
that needs deeper grounding.

**Next session**: Covariance matrix deep-dive — eigenvalues,
condition number, Mahalanobis distance — then re-read Section 3
of the paper with that grounding.

---

## 06/05/2026 (Wed) — Project repo created

Created GitHub repository `kappa-kinisi`. Set up local folder
structure (code, experiments/exp_a_length, notebooks, notes).
Established working format for daily project log.

---


## 27/03/2026 to 06/05/2026 — Initial environment setup

Set up kinisi locally and confirmed end-to-end pipeline works:
- Installed kinisi, MDAnalysis, scipp, numpy, scipy, matplotlib
- Ran initial 3D lattice random walk simulations
- Confirmed D* recovered close to 1.0 across multiple seeds
- Started preliminary plotting and diagnostic scripts

---



## 27/03/2026 (Fri) — Second supervisor meeting

Discussion with Andrew on the major aspects suspected to drive κ
upwards. 

---

## 05/03/2026 to 27/03/2026 — Background reading and material study

**Reading list worked through:**

- Reference paper: McCluskey, Coles & Morgan,
  J. Chem. Theory Comput. 2025, 21, 79–87.
  https://doi.org/10.1021/acs.jctc.4c01249
- YouTube playlist on diffusion / MSD analysis:
  https://www.youtube.com/watch?v=Je4bU3g_QGk&list=PLyHAvCibkccQEFYXdM6r8WG4GQULRKmRA
- kinisi documentation:
  https://kinisi.readthedocs.io/en/latest/
- kinisi GitHub repository:
  https://github.com/kinisi-dev/kinisi
- Specific code reference — line 241 of diffusion.py
  (covariance matrix construction site):
  https://github.com/kinisi-dev/kinisi/blob/1861ba0dacda882b1be2e42c4262b66f842478e9/kinisi/diffusion.py#L241

**Outcome**: developed working understanding of the paper's method
(Bayesian regression with multivariate normal likelihood), the
analytical Σ' construction (Eqs. 6 and 7), and where the condition
number problem enters (likelihood evaluation in Eq. 5 requires Σ'⁻¹).

---


## 05/03/2026 (Thu) — First supervisor meeting

First meeting with supervisor Andrew McCluskey. Project framing
established: investigate why the condition number k of the
covariance matrix Σ' in kinisi gets large, beyond the band-aid
reconditioning fix described in the paper.

---
## 03/03/2026 (Tue) — Project assigned

Project assigned: Investigations into the Numerical Stability of
Covariance Matrix Inversions in the kinisi library.


 
