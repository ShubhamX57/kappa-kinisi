# Running Guide

*A step-by-step guide to setting up and running the code for*
*Investigations into the Numerical Stability of Covariance Matrix Inversions.*

This guide assumes no prior familiarity with the project. If anything does not work, the
[Troubleshooting](#troubleshooting) section at the end lists every error I have seen and its fix.

---

## 1. What is in the submission

```
Application_code/
├── data/               all result files and inputs - nothing needs downloading
├── notebooks/          the seven analysis notebooks
├── python_scripts/     the fourteen analysis scripts, plus shared helpers
└── plots/              output folder for any figures the code writes
```

The `data/` folder is included in full, so **every file can be run without downloading anything**.
The code finds this folder automatically, wherever you launch it from - see step 4.

---

## 2. Setting up the environment

The code needs Python 3.11 and a fixed version of the `kinisi` package. The version matters: the
analysis rebuilds an internal matrix from what `kinisi` stores, and that storage changed between
releases, so a different version will not give the same results.

### Option A - conda (recommended)

```bash
conda env create -f environment.yml
conda activate kappa-kinisi
```

### Option B - pip and a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Both install the same packages. To confirm it worked:

```bash
python -c "import kinisi, numpy, scipp; print('environment ready')"
```

If that prints `environment ready`, you are set.

---

## 3. Where the data lives, and how the code finds it

Every result file is already in the `data/` folder inside the submission. **You do not need to
move it, copy it, or download anything.**

The code locates `data/` by searching upward from wherever it is launched until it finds that
folder. This means you can run a script or notebook from the project root, from inside
`python_scripts/`, or from inside `notebooks/`, and it will always find the data. The only
requirement is that you launch from *somewhere inside* the `Application_code` folder.


---

## 4. Running the scripts

Open a terminal in the `Application_code` folder with the environment activated, and run any script
by name:

```bash
python python_scripts/reconstruction.py
```

Each script prints its results to the terminal and, where relevant, saves a file into `data/`.

**Because all the data is provided, every script runs in seconds to minutes.** A script that would
otherwise take hours to compute its input simply loads the finished file instead.

### The recommended first run

To confirm everything is working, start with the two quickest scripts. Neither takes more than a
few seconds:

```bash
python python_scripts/verify_data.py     # checks every data file is present and intact
python python_scripts/reconstruction.py  # reproduces a headline result
```

`verify_data.py` prints a line for each file in `data/`; if it runs without error, the data is
sound. `reconstruction.py` demonstrates the central technical point of the project - that the
matrix which caused the failure can be rebuilt from what the package keeps.

### What each script does

| script | what it shows | runtime with data |
|---|---|---|
| `verify_data.py` | every data file is present and readable | seconds |
| `reconstruction.py` | the raw covariance can be rebuilt; the injection is faithful | seconds |
| `regenerate_population_v2.py` | the 64,000-fit population; the fourteen affected cases | seconds* |
| `surgical_swap_test.py` | replacing one direction of 126 repairs every affected fit | seconds |
| `run_anomaly_harness.py` | recovery under the spectral treatments | seconds |
| `realfit_c_sweep.py` | how the adaptive-floor coefficient was chosen | seconds |
| `variance_repair_test.py` | recovery under the variance treatments | seconds |
| `shrink_floor_test.py` | the four treatments compared side by side | seconds |
| `shrink_validation.py` | the treatment does not distort healthy fits | seconds* |
| `bootstrap_validation.py` | why the resampled covariance is misleading | seconds |
| `detector_specificity.py` | the detection check's false-alarm rate | seconds |
| `nonreversal_recovery_study.py` | the correction works on a different system | seconds* |
| `persistent_shape_check.py` | the treatment's assumption under fast motion | seconds |
| `true_variance_test.py` | the true uncertainty, measured directly | seconds |

*These scripts *generate* their data from scratch if the file is absent, which takes much longer
(up to eight hours for the population). With the provided `data/` folder they load the finished
file and return at once. **Leave the provided files in place to keep them fast.**

---

## 5. Running the notebooks

From inside the `notebooks/` folder, with the environment activated:

```bash
cd notebooks
jupyter lab
```

Then open a notebook and run its cells from the top (**Kernel >> Restart and Run All** runs the
whole thing). Each notebook explains what every cell does before it runs and what the result means
afterwards.

Start with **`Anomalous_D_Value_Inspect_full.ipynb`** - it is the main notebook and covers the
whole investigation in order.

The other six each cover one part:

| notebook | topic |
|---|---|
| `Anomalous_B_Value_Inspect_full.ipynb` | where the failure was first noticed |
| `Suspect_A_Testing.ipynb` | ruling out window overlap as the cause |
| `Suspect_B_Testing.ipynb` | confirming sampling noise as the cause |
| `model-nodel.ipynb` | comparing the modelled and measured matrices |
| `flooring.ipynb` | surveying ways to repair the matrix |
| `method_comparison.ipynb` | comparing repair and inversion methods |

All notebooks read from the provided `data/` folder, so they run without any preparation beyond
activating the environment.

---

## 6. Running the tests (optional)

A small test suite checks the properties the results rely upon. It needs no data and runs in under
a second:

```bash
python -m pytest tests/ -v
```

All eleven tests should pass.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'kinisi'` (or numpy, scipp, …)**
The environment is not active. Run `conda activate kappa-kinisi`, or
`source .venv/bin/activate`, and try again.

**`FileNotFoundError` ending in a `.npy` or `.npz` name**
The code could not find the `data/` folder. Make sure you are running from somewhere *inside* the
`Application_code` folder, and that `data/` is still there with its files in it. Running
`python python_scripts/verify_data.py` will confirm what the code can see.

**A script seems to hang for a long time**
One of the starred scripts is regenerating its data because the file is missing. Stop it
(Ctrl-C), check that the file it needs is present in `data/`, and run it again - it should then
finish in seconds. The files most worth checking are `failure_population_v2.npy` and
`all_bad_seeds_v2.npy`.

**A notebook cell fails with a name that is not defined**
The cells must be run in order. Use **Kernel >> Restart and Run All** rather than running cells
individually.

**`kinisi` behaves unexpectedly or gives different numbers**
Check the version: `python -c "import kinisi; print(kinisi.__version__)"` should print `2.0.5`.
A different version will not reproduce these results, for the reason given in step 2.

**The results print but no figure appears**
The scripts and notebooks are written to report their findings as numbers and, in the notebooks, as
plots. The standalone figure-generation code is not part of this submission; the numbers printed
are the results themselves.

---

## A note on how the analysis works, for context

Three facts underpin everything, and knowing them makes the code easier to follow.

The package stores only the covariance matrix *after* its own repair has been applied, so the
matrix that actually caused a failure is no longer available and has to be **rebuilt from the
stored numbers**. This reconstruction is what `reconstruction.py` verifies.

A repaired matrix cannot simply be handed to the fitting routine, because that routine rebuilds the
covariance internally and would ignore it. Every experiment therefore **replaces the routine's
matrix-building step** so that the repaired matrix is actually used.

Finally, two number scales appear in the output. The package's own scale reports a healthy result
near 9,980; the experiments report a rescaled value near 1. They describe the same thing four
orders of magnitude apart, so a result near 10,000 and a result near 1 can both be correct.