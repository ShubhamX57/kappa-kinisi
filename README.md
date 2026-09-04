# kappa-kinisi

## Investigation of a rare silent failure in Bayesian MSD analysis

This repository contains the computational work supporting the MSc Scientific Computing with Data Science dissertation:

> **An Investigation into a Rare Silent Failure in Bayesian Mean-Squared-Displacement Analysis**

The project investigates a rare failure in Bayesian diffusion analysis using `kinisi`. The main study uses controlled three-dimensional lattice random walks with a known diffusion coefficient, then tests candidate mechanisms, covariance diagnostics, counterfactual interventions, detection methods, correction strategies, calibration, generalisation, and real molecular-dynamics examples.

The project is organised so that a future researcher can reproduce the analysis, inspect the exploratory notebooks, run individual analysis scripts, and access the saved result/data files.

---

## Repository structure

```text
kappa-kinisi/
├── code/
│   ├── notebooks/
│   │   ├── Anomalous_D_Value_Inspect_full.ipynb
│   │   ├── Suspect_A_Testing.ipynb
│   │   ├── Suspect_B_Testing.ipynb
│   │   ├── flooring.ipynb
│   │   ├── method_comparison.ipynb
│   │   ├── descriptive.ipynb
│   │   ├── model-nodel.ipynb
│   │   └── ...
│   │
│   └── python_scripts/
│       ├── bootstrap_validation.py
│       ├── detector_specificity.py
│       ├── nonreversal_recovery_study.py
│       ├── persistent_shape_check.py
│       ├── realfit_c_sweep.py
│       ├── reconstruction.py
│       ├── regenerate_population_v2.py
│       ├── run_anomaly_harness.py
│       ├── shrink_floor_test.py
│       ├── shrink_validation.py
│       ├── surgical_swap_test.py
│       ├── true_variance_test.py
│       ├── variance_repair_test.py
│       ├── verify_data.py
│       ├── kinisi_helpers.py
│       ├── make_all_figures.py
│       ├── make_diagrams.py
│       ├── make_si_figures.py
│       └── plot_style.py
│
├── data/
│   ├── population and anomaly result files
│   ├── correction/calibration results
│   ├── real-data inputs
│   └── large `.npz` trajectory files
│
├── plots/
│   └── generated figures
│
└── README.md
```

The exact contents of `notebooks/`, `python_scripts/`, `data/`, and `plots/` may change as the project is extended. The important distinction is:

- **`code/notebooks/`**: interactive investigation and development notebooks.
- **`code/python_scripts/`**: reusable analysis, validation, reconstruction and figure-generation scripts.
- **`data/`**: saved inputs, checkpoints and numerical outputs.
- **`plots/`**: generated figures used by the dissertation and supporting material.

---

# 1. Software requirements

The main analysis was performed using:

- **Python 3.11**
- **kinisi 2.0.5**
- NumPy
- SciPy
- MDAnalysis
- pymatgen
- scipp
- matplotlib
- SciencePlots
- statsmodels
- tqdm
- emcee

The report records Python 3.11 and `kinisi` 2.0.5 as the core computational environment.

Because some experiments depend on the exact numerical behaviour of the software stack, using the same Python/`kinisi` versions is recommended when reproducing dissertation results.

---

# 2. Create the environment

A virtual environment can be created with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the required packages:

```bash
pip install \
    kinisi==2.0.5 \
    numpy \
    scipy \
    MDAnalysis \
    pymatgen \
    scipp \
    matplotlib \
    SciencePlots \
    statsmodels \
    tqdm \
    emcee
```

Check the critical versions:

```bash
python --version
python -c "import kinisi; print(kinisi.__version__)"
```

Expected:

```text
Python 3.11.x
2.0.5
```

If an environment specification is supplied separately with the project, prefer that specification over installing unpinned packages.

---

# 3. Getting the repository

Clone the repository:

```bash
git clone https://github.com/ShubhamX57/kappa-kinisi.git
cd kappa-kinisi
```

If the repository uses Git LFS for the large data files, install Git LFS before downloading the data:

```bash
git lfs install
git lfs pull
```

Check whether the large files are present:

```bash
ls -lh data/
```

The largest trajectory files are several hundred MB, so they are handled separately from ordinary Git blobs.

---

# 4. Accessing the data

The `data/` directory contains both raw inputs and saved results from the computational experiments.

Typical groups of files include:

### Population study

```text
failure_population*.npy
rw_*records*.npy
all_bad_seeds*.npy
```

These contain the large random-walk population results and the identified anomalous seeds.

### Diagnostic and mechanism tests

```text
eigen_swap_test.npy
negatives_vs_atoms.npy
surgical_swap_test.npy
```

These contain outputs from eigenvalue/eigenvector, sampling and intervention experiments.

### Correction and calibration tests

```text
shrink_floor_test.npy
variance_repair_test.npy
shrink_coverage.npy
bootstrap_coverage.npy
tau_sweep.npy
kinisi_vs_shrink.npy
```

These contain results used to compare correction strategies and assess calibration.

### Generalisation tests

```text
nonreversal_population.npy
nonreversal_bad_seeds.npy
nonreversal_repair.npy
```

### Large molecular-dynamics data

```text
kinisi_rw_data_1.npz
model.npz
no_model.npz
```

These are large input datasets and may require Git LFS.

### Archive files

Files under:

```text
data/archive/
```

are retained as historical/intermediate project material where applicable. They are not necessarily required for reproducing the principal dissertation results.

---

# 5. Running the notebooks

The notebooks document the development and investigation of the project.

Start Jupyter from the repository root:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Open:

```text
code/notebooks/
```

The principal notebook is:

```text
Anomalous_D_Value_Inspect_full.ipynb
```

The main hypothesis-specific notebook is:

```text
Suspect_A_Testing.ipynb
```

Other notebooks investigate individual correction, descriptive, flooring and comparison experiments.

## Python helper modules used by notebooks

Some notebooks import modules from:

```text
code/python_scripts/
```

For example:

```python
from kinisi_helpers import build_analyzer
from plot_style import use_style
```

If Python cannot find these modules because the notebook is being run from `code/notebooks/`, add the script directory to `sys.path` before importing:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd().parent / "python_scripts"))
```

Alternatively, set the module path before launching Jupyter:

```bash
export PYTHONPATH="$PWD/code/python_scripts:$PYTHONPATH"
jupyter lab
```

---

# 6. Running the Python analysis scripts

Move into the script directory:

```bash
cd code/python_scripts
```

The scripts can then be run individually with:

```bash
python script_name.py
```

For example:

```bash
python regenerate_population_v2.py
```

or:

```bash
python variance_repair_test.py
```

The script names indicate their role:

| Script | Purpose |
|---|---|
| `regenerate_population_v2.py` | Regenerate/check the main controlled population |
| `run_anomaly_harness.py` | Run the anomaly analysis harness |
| `reconstruction.py` | Reconstruct and compare covariance matrices |
| `verify_data.py` | Check saved data and analysis inputs |
| `detector_specificity.py` | Evaluate the Bayesian/OLS detector |
| `nonreversal_recovery_study.py` | Test generalisation to correlated-step walks |
| `persistent_shape_check.py` | Examine persistent/ballistic behaviour |
| `shrink_validation.py` | Validate variance shrinkage |
| `shrink_floor_test.py` | Compare shrinkage and flooring |
| `variance_repair_test.py` | Test variance-profile repair |
| `true_variance_test.py` | Compare estimated variance with an independently measured reference |
| `bootstrap_validation.py` | Test bootstrap-based variance estimation |
| `surgical_swap_test.py` | Test local eigenvector intervention |
| `realfit_c_sweep.py` | Evaluate correction behaviour over coefficient values |

Not every script is required for a routine reproduction of the main dissertation figures. Some scripts correspond to exploratory, validation or follow-up experiments.

---

# 7. Figure generation

Figure-generation utilities are kept in:

```text
code/python_scripts/
```

Important files include:

```text
make_all_figures.py
make_diagrams.py
make_si_figures.py
plot_style.py
```

The plotting style is centralised in:

```text
plot_style.py
```

so that figures use consistent dimensions, typography, colours, labels and export settings.

Generated figures are written to:

```text
plots/
```

The scripts generally save both:

```text
.pdf
.png
```

The PDF version should be preferred when figures are inserted into the dissertation because it preserves vector graphics.

---

# 8. Reproducing the main population study

The main controlled study contains:

```text
64,000 fits
16,000 fits at 16 atoms
16,000 fits at 24 atoms
16,000 fits at 32 atoms
16,000 fits at 48 atoms
```

The random-walk system has:

```text
128 frames
126 lag values
2–128 s lag range
1 s time step
```

The walk is constructed so that the true diffusion coefficient is known analytically.

The population-generation script writes checkpointed results so that later analyses can operate on the saved results rather than silently regenerating the population.

For a fresh regeneration, run the relevant population script from:

```bash
cd code/python_scripts
python regenerate_population_v2.py
```

This can be computationally expensive because the original study involved a large number of Bayesian fits.

---

# 9. Reproducing the anomaly analysis

The principal failure study identifies the anomalous cases by comparing the returned diffusion coefficient with the healthy population.

The dissertation reports:

```text
14 affected cases
out of
64,000 controlled fits
```

The identified seeds are stored in the population/anomaly result files under `data/`.

The anomaly analysis can therefore normally be reproduced from the saved checkpoint/result files without rerunning the entire 64,000-fit population.

---

# 10. Important distinction: raw versus treated covariance

A central part of the project is the distinction between:

1. the **raw reconstructed covariance**, and
2. the covariance after kinisi's internal treatment.

The raw covariance is reconstructed from the stored variance profile and sample counts so that the failure can be studied before the package's internal treatment.

The relevant helper is:

```text
code/python_scripts/kinisi_helpers.py
```

The dissertation verifies the reconstruction against matrices retained by `kinisi`.

This distinction is important when interpreting results. A result based on the treated covariance is not automatically a result about the raw covariance that generated the failure.

---

# 11. Counterfactual and correction experiments

Several experiments modify covariance components and rerun the genuine Bayesian fit.

These include:

- eigenvalue/eigenvector substitutions;
- single-direction eigenvector interventions;
- adaptive eigenvalue flooring;
- variance-profile reconstruction;
- variance shrinkage;
- resampled covariance;
- existing kinisi reconditioning.

The important methodological point is that the project does not rely only on a least-squares proxy for these tests. Treated covariance matrices are injected into the Bayesian analysis pathway so that the actual likelihood is evaluated.

This is one reason why the helper/injection code should be kept together and run using the same environment.

---

# 12. Real molecular-dynamics data

The project also examines real molecular-dynamics examples distributed with `kinisi`.

These experiments are intended to establish whether the numerical behaviour observed in the controlled system is present outside the random-walk test system.

They do **not** provide ground truth for the real material diffusion coefficient, so they are not used as accuracy benchmarks in the same way as the controlled random-walk population.

---

# 13. Reproducibility and saved results

The principal computational experiments were checkpointed during execution.

This means that most analysis scripts should read previously saved results rather than silently refitting the entire population.

When reproducing a result, check first whether the corresponding `.npy` result already exists in:

```text
data/
```

This is especially important for the population-scale experiments because rerunning them unnecessarily can be expensive.

---

# 14. Expected scientific workflow

A useful way to understand the project is:

```text
Controlled random walks
        ↓
64,000 Bayesian fits
        ↓
Identify rare anomalous cases
        ↓
Reconstruct raw covariance
        ↓
Test competing explanations
        ↓
Test covariance diagnostics
        ↓
Investigate eigenvector geometry
        ↓
Counterfactual interventions
        ↓
Develop Bayesian–OLS consistency check
        ↓
Evaluate correction strategies
        ↓
Calibration / healthy controls
        ↓
Generalisation tests
        ↓
Real molecular-dynamics examples
```

The repository is organised to mirror this workflow.

---

# 15. Reproducibility checklist

Before treating a reproduction as comparable to the dissertation results, check:

```text
[ ] Python 3.11
[ ] kinisi 2.0.5
[ ] Required scientific Python packages installed
[ ] Correct branch/version of the repository checked out
[ ] Large LFS data downloaded
[ ] data/ directory available
[ ] code/python_scripts available on Python path
[ ] Correct random seeds used
[ ] Existing checkpoint files not accidentally overwritten
[ ] Figures generated from the intended saved result files
```

For publication figures, use the generated PDF files rather than screenshots of notebook output.

---

# 16. Branches

The repository may contain analysis branches corresponding to different stages or hypothesis-specific investigations.

Examples include:

```text
main
suspect-a-testing
suspect-b-testing
```

The `main` branch should be treated as the primary integrated version.

The hypothesis-specific branches preserve focused analysis that contributed to the investigation.

When comparing results across branches, always check:

```bash
git branch
git log --oneline --max-count=10
```

and record the commit used for the result.

---

# 17. Troubleshooting

### `ModuleNotFoundError: kinisi_helpers`

Make sure `code/python_scripts/` is on the Python path:

```bash
export PYTHONPATH="$PWD/code/python_scripts:$PYTHONPATH"
```

or add the path from inside the notebook:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd().parent / "python_scripts"))
```

### `FileNotFoundError` for a saved dataset

Check that the notebook/script is being run from the expected directory and that the `data/` directory exists:

```bash
ls data/
```

Avoid changing relative paths until you have checked the current working directory:

```python
from pathlib import Path
print(Path.cwd())
```

### Git reports that `data/` is ignored

Some data may be intentionally excluded from ordinary Git tracking. For files that are deliberately versioned:

```bash
git add -f data/<filename>
```

Large files should use Git LFS where configured.

### Git LFS command is unavailable

Install Git LFS first:

```bash
brew install git-lfs
git lfs install
```

Then retrieve the large objects:

```bash
git lfs pull
```

### A script fails because of an old notebook state

Rebuild the analyser/process from scratch rather than reusing an object that may have been modified or injected during another experiment. Several validation steps in this project depend on avoiding state leakage between experiments.

---

# 18. Code documentation

The code is intended to be readable by someone who did not perform the original investigation.

Reusable functions should contain docstrings describing:

- what the function does;
- the main inputs;
- returned values;
- important assumptions.

Analysis scripts should also make clear which scientific question they address.

The notebook documentation is more explanatory, while the standalone scripts should remain focused on executable analysis.

---

# 19. Relationship to the dissertation

The dissertation is the authoritative source for the scientific interpretation of the results.

This repository provides the computational record behind it.

The main report analyses include:

- frequency and reproducibility of the failure;
- covariance reconstruction;
- competing mechanism tests;
- spectral diagnostics;
- eigenvector localisation;
- counterfactual eigenvector intervention;
- Bayesian/OLS detection;
- correction comparisons;
- calibration;
- generalisation;
- real molecular-dynamics data.

The supplementary material provides additional case-by-case and verification information.

---

# 20. Citation and reuse

If you use the code or results from this repository, please cite the associated dissertation and the relevant software/literature sources listed in the dissertation bibliography.

The repository is intended to support reproducibility and future research. Before reusing a numerical result, check the script, saved input/result file, branch and commit from which it was produced.

---

## Quick start

For someone returning to the project, the shortest route is:

```bash
git clone https://github.com/ShubhamX57/kappa-kinisi.git
cd kappa-kinisi

git lfs install
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate

pip install \
    kinisi==2.0.5 \
    numpy scipy MDAnalysis pymatgen scipp \
    matplotlib SciencePlots statsmodels tqdm emcee

export PYTHONPATH="$PWD/code/python_scripts:$PYTHONPATH"

jupyter lab
```

Then open:

```text
code/notebooks/Anomalous_D_Value_Inspect_full.ipynb
```

For standalone analysis:

```bash
cd code/python_scripts
python <script_name>.py
```

---

## Project status

The repository contains the computational analysis supporting the submitted MSc dissertation. Some files are intermediate or exploratory by design. The principal saved results and documented scripts should be preferred over rerunning exploratory development notebooks when reproducing the reported figures and tables.
