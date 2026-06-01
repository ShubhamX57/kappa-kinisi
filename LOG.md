# Project log : kappa-kinisi|


--- 
## 01/06/2026 (Mon) - Met with supervisor (weekly meeting)

**Did:** Discussed Suspect A and its timestep correlation with the Andrew. 
Developed a weekly plan for conducting tests related to timestep analysis and validation. 
Agreed to prepare a weekly presentation summarizing progress and findings to ensure that important 
information is documented and retained between meetings.


---
## 31/05/2026 (Sun) - Working on Equations 

**Did:** Solved equations for Timestep correlation.

---
## 30/05/2026 (Sat) - Working on Suspect A.

**Did:** Working on Adjacent timesteps share data >> similar rows.

---
## 29/05/2026 (Fri) - Reading the paper Won et al. 2013

**Did:** Reading the Won et al. 2013 on condition number regularized covariance estimation. 
Foucing on condition number, convex optimization, covariance estimation, cross-validation and eigenvalue.


---
## 28/05/2026 (Thu) - Meeting with Andrew

**Did:** Set up admin access on GitHub and Slack. Agreed weekly catch-ups
every Monday. Discussed the four suspects for the large condition number
(A: timestep correlation, B: noisy variance at long times, C: matrix size,
D: floating-point cancellation). Noted B and D are the same problem
at two levels.

---
## 27/05/2026 (Wed) — Reading on condition numbers

**Did:** Looked up some background reading on condition numbers and
covariance matrices to understand where the squash idea fits.

---
## 26/05/2026 (Tue) — emcee notes

**Did:** Made notes on Foreman-Mackey. Affine invariance and
autocorrelation time are the two ideas to remember.

---
## 25/05/2026 (Mon) - emcee paper (Foreman-Mackey et al. 2019)

**Did:** Read the emcee paper - the MCMC sampler kinisi uses to sample
p(D*|x). Closes my mental picture of the kinisi pipeline: I now understand
both the construction (upstream of likelihood) and the MCMC sampling
(downstream of likelihood). Key concepts: affine-invariant ensemble
sampler, 32 walkers, autocorrelation-based convergence diagnostics.

---

## 22/05/2026 (Fri) - Resumed Phase 1 reading

**Did:** Back to the paper after the portfolio break. Started T1.2 —
annotating Eq. 5 (the log-likelihood) in my own words, term by term.

---
## 21/05/2026 (Thu) - Weitten some notes on floating-point 
**Did:** Made a note on floating-point error and watched video to understand logic behide it.

---
## 20/05/2026 (Wed) — Portfolio submitted + FP article review

**Did:** Submitted portfolio via Blackboard before midday deadline.
Read Julia Evans' floating-point article (Cameron's suggestion).
Example 3 (catastrophic cancellation in variance) maps directly onto
Eq. 7 of the Andrew's paper - gives a fourth candidate root cause
for large k alongside (A) correlation, (B) sample noise, (C) matrix size.

**Next session:** Resume T1.2 (annotate Eq. 5). Write up suspect D in
notes. Raise FP hypothesis with Andrew.

---
## 19/05/2026 (Tue) – Combined Portfolio Documentation  

**Did:** Developed and refined the portfolio documentation, progressing from draft versions to a polished submission for Assessment Brief 2.


---
## 18/05/2026 (Mon) - Portfolio Meeting with Project Assessor Dr. Cameron Beevers

**Did:** Discussed the project portfolio, with focus on the covariance matrix and floating-point precision. Dr Beevers performed small example runs that demonstrated how floating-point errors may be the underlying issue behind the increasing number of *k*.

**Next session:** Review the article suggested by Dr. Beevers on floating-point problems for further insights:
https://jvns.ca/blog/2023/01/13/examples-of-floating-point-problems/

---

## 15/05/2026 (Fri) — Branch tidy-up

**Did**: Moved covariance plotting work onto a `basic-testing` branch
to keep `main` clean. Practised handling a few git tangles along the
way (accidentally committed to main, force-push to fix it, sorting
untracked files when switching branches).

--- 

## 14/05/2026 (Thu) — Covariance plotting script

**Did**: Wrote code/covariance_basics.py — a simple script that
generates three plots: a correlated data scatter, the covariance
matrix as a heatmap, and a demo of how the condition number grows
as data is squashed. Ran it and saved the plots.

**Learned**: Confirmed visually that a thinner data cloud gives a
much bigger condition number — squashing the data 10x makes the
condition number roughly 100x larger. This is the same mechanism
behind kinisi's instability: a near-singular covariance matrix
amplifies errors when inverted.


---
## 13/05/2026 (Wed) — Covariance

**Did**: Worked on covariance matrix material. Did some sample plotting using matplotlib to get overview of covariance matrix.

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


 
