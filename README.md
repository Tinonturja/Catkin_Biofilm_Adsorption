[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#)

# Catkin Biofilm Adsorption — Modeling Code

Code accompanying an in-preparation manuscript on reactive-dye adsorption by
a citric-acid crosslinked PVA/TiO2/cellulose-microcrystal (CMC) biofilm, in
which the CMC is extracted for the first time from the flower fibre of
*Saccharum spontaneum* (Catkin), an abundant, unexploited cellulose source.
The film is evaluated against the anionic reactive dye Avitera Light Red SE.
This is a companion project to [WCF_Biofilm_Adsorption](https://github.com/Tinonturja/WCF_Biofilm_Adsorption),
which shares the same fabrication/characterization protocol and modeling
approach, applied to a waste-cotton-derived CMC instead.

Full derivation, experimental conditions and figures are in the accompanying
manuscript (`manuscript/`).

## Key results

| Quantity | Value |
| :--- | :--- |
| Kinetics — PSO fit | R² = 0.966, qₑ = 1.158 mg/g (vs. PFO R² = 0.919) |
| Isotherm — Freundlich fit | R² = 0.997, n = 1.182, Kf = 0.0368 (Langmuir ruled out — negative Kₗ) |
| Dye removal | peaks ~55% at 110 min, ~53% at 170 min equilibrium; 37–40% across 20–60 ppm |
| UV-Vis band gap (Tauc) | 4.8 eV |
| PINN3D — in-sample fit | kinetics R² = 0.99995, isotherm R² = 0.99999 |
| PINN3D — leave-one-out CV (honest generalization) | overall R² = 0.538 (kinetics arm R² = 0.739, isotherm arm R² = 0.014) |

The large gap between the in-sample and leave-one-out numbers is reported
deliberately, not hidden — see the manuscript's Results section. It mirrors
a similar isotherm-arm instability reported in the companion WCF paper on
its own small (n=5) isotherm dataset.

## Installation

Two equivalent ways to install — pick one.

**As an editable package (recommended if you'll import the modules elsewhere):**
```bash
git clone https://github.com/Tinonturja/Catkin_Biofilm_Adsorption.git
cd Catkin_Biofilm_Adsorption
pip install -e .
```

**Pinned, reproducible environment (exact versions used to generate every figure and result):**
```bash
git clone https://github.com/Tinonturja/Catkin_Biofilm_Adsorption.git
cd Catkin_Biofilm_Adsorption
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verified on Python 3.10. `torch` (CPU build) is required for all the PINN
scripts; PINN3D training is run on CPU deliberately (see the note in
`src/pinn3D/train.py` — the physics-loss double-backward pass produces NaN
gradients on PyTorch's MPS backend for some inputs).

## Structure

```
src/
├── pinn1D/                    # earlier, single-input (1-D) PINN models — superseded by pinn3D below, kept for history
└── pinn3D/                    # current approach: a single PINN jointly fitting PSO kinetics + Freundlich isotherm
    ├── data.py                 # loads and prepares the combined kinetics/isotherm dataset
    ├── models.py                # PINN3DModel definition
    ├── train.py                  # main training script (fixed loss weights, N-bound reparameterization)
    ├── loo_cv_pinn3d.py            # leave-one-out cross-validation driver
    ├── plotting.py / plot_style.py  # figure generation (Nature-journal-style aesthetic)
    └── make_equations.py             # renders the loss-function equation block figure

data/                          # raw experimental data (single Excel workbook, kinetics + isotherm sheets)
figures/                       # generated figures referenced by the manuscript
results/                       # model outputs: fitted parameters, training curves, LOO-CV results
models/                        # trained PINN checkpoints (small — kept in version control, unlike WCF's)
manuscript/                    # current manuscript draft; manuscript/archive/ holds superseded revisions
docs/                          # supporting notes: PINN3D methodology writeup, paper-asset tracking, notebook
tests/                         # unit tests
```

## Reproducing results

```bash
python -m pinn3D.train        # trains PINN3D, writes results/pinn3d_training_results.csv + results/pinn3d_parameters.md
python -m pinn3D.loo_cv_pinn3d  # leave-one-out cross-validation (14 folds, ~25 min on CPU)
```

## Data

Experimental kinetics (`n=9`, 0–170 min) and isotherm (`n=5`, 20–60 ppm)
measurements are in `data/data_of_biofilm.xlsx`. Single-shot values, no
replicates — this is the full dataset described in the manuscript's Methods
section; there is no additional held-out data.

## Note on `models/` and `results/`

Unlike the companion WCF repo, PINN3D's checkpoints are small (~10 KB each,
32 hidden units) and are kept in version control directly rather than
excluded via `.gitignore` — no Git LFS needed here.

## Status

This repository is being prepared ahead of manuscript submission. A DOI via
Zenodo will be issued once a first tagged release is cut; the citation
details in `CITATION.cff` will be updated at that point.

## Citation

See `CITATION.cff`. A citation for the accompanying paper will be added once
it is submitted.

## License

MIT — see `LICENSE`.
