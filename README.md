# Catkin Biofilm Adsorption: kinetic and isotherm analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23195584.svg)](https://doi.org/10.5281/zenodo.23195584)

Analysis code and outputs for a study of reactive dye (Avitera Light Red SE) adsorption on a
citric-acid crosslinked PVA/TiO2/cellulose-microcrystal biofilm, in which the cellulose
microcrystals were extracted from the flower fibre of *Saccharum spontaneum* (catkin).

This repository covers the modelling part of the study only: validation of the dataset,
classical kinetic and isotherm fits, and a simulation of the sampling design. The experimental work is described in the accompanying manuscript.

## Data

`data/data_of_biofilm.xlsx` holds the measurements. The scripts read two sheets:

- `catkin (pfo pso)`: kinetic series, 8 measured points from 10 to 170 min at one initial
  concentration (40 mg/L) and one dose. The t = 0 row is q = 0 by definition and is not
  treated as a measurement.
- `catkin`: isotherm series, 5 points at initial concentrations of 20 to 60 mg/L, one
  contact time (170 min) and one dose.

`data/calibration_standards.csv` holds the six calibration standards (10 to 100 mg/L). A straight line
through them reproduces the calibration used in the workbook.

All values are single measurements without replicates.

## Main results

Kinetics, nonlinear least squares on the 8 measured points
(`results/final_analysis/kinetic_candidates.csv`):

| Law | Parameters | RMSE (mg/g) | Leave-one-out RMSE | R2, measured points | Akaike weight |
|---|---|---|---|---|---|
| Burst + sqrt(t) | q_burst 0.597, k_id 0.0461 | 0.025 | 0.033 | 0.975 | 0.77 |
| Power law | a 0.473, b 0.177 | 0.030 | 0.037 | 0.964 | 0.18 |
| Elovich | alpha 1.158, beta 6.131 | 0.036 | 0.046 | 0.947 | 0.04 |
| Pseudo-second-order | qe 1.158, k2 0.1093 | 0.066 | 0.094 | 0.821 | below 0.001 |
| Pseudo-first-order | qe 1.055, k1 0.0864 | 0.102 | 0.148 | 0.576 | below 0.001 |

Akaike weights are computed over all eleven candidate laws. Parameter values are on the scale of the
q column in the workbook (see Limitations).

Isotherm, 5 points (`results/final_analysis/isotherm_fits.csv`):

| Law | Parameters | RMSE vs Ce (mg/g) |
|---|---|---|
| Freundlich, q = Kf Ce^n | Kf 0.0334, n 1.212 (95 % profile interval 1.06 to 1.38) | 0.037 |
| Linear | K 0.0677 | 0.102 |
| Langmuir | degenerate, collapses to the linear law | 0.102 |

The interval above holds only at the workbook calibration. The exponent depends on the calibration line:
it is 1.21 with the fitted intercept, 1.02 with a line forced through the origin, and 1.44 when the highest
standard is dropped. Propagating the uncertainty of the calibration line gives a 95 % range of 0.56 to 2.23
(`results/final_analysis/isotherm_calibration_sensitivity.csv`).

What these support:

- Pseudo-first-order and pseudo-second-order models do not describe the kinetic series. Their
  R2 values rise to 0.92 and 0.97 only when the definitional (0, 0) row is counted as data.
- Uptake has a fast part complete within 10 min (61 % of the 170-min value) and a slow part
  that has not levelled off at 170 min. Three two-parameter descriptions of the slow part fit
  within 0.025 to 0.036 mg/g and cannot be separated with eight points.
- The burst + sqrt(t) form is an empirical description of the measured window. It is not
  evidence of intraparticle diffusion control.
- Uptake at 170 min is close to proportional to concentration over 20 to 60 mg/L (removal 37 to 41 %).
  Whether it bends upward cannot be decided, because the Freundlich exponent is not distinguishable
  from 1 once calibration uncertainty is included.
- No equilibrium capacity and no maximum adsorption capacity can be reported from these data.

## Limitations

- Eight kinetic points at one concentration and one dose, and five isotherm points at one
  contact time and one dose. Dosage takes one value per arm, so no dosage effect can be
  identified.
- The solution-to-film ratio of the kinetic run is not stored in the workbook. It is taken as 100 mL
  of 40 mg/L dye on 1.2 g of film (1/12 L/g, 12 g/L) and set in `src/final_analysis/common.py`. The
  kinetic laws are fitted to the q column directly, so the ratio enters only the checks that link q
  to concentration. A different ratio would rescale qe-type parameters and rate constants by a
  constant factor and would not change the ranking of the kinetic laws or any R2.
- No tested ratio lets a single equilibrium law describe both arms, so they are modelled separately
  and the isotherm is read as an uptake curve at 170 min. See `results/final_analysis/classical_fits.txt`,
  part C.
- The calibration line has a residual of about 3 mg/L, which is larger than the fit residuals. It is
  shared by every point, so point-wise cross-validation understates the real uncertainty.
- The catkin column of the `removal% against time` sheet is not consistent with the q(t)
  series. The q(t) series is used as the reference and that column is not used in any model.
- The design-density study is a simulation under stated noise assumptions. Its synthetic data
  are not measurements.

## Repository layout

```
data/                         experimental workbook and calibration standards
src/figstyle.py               one figure style shared by every plot (fonts, sizes, colours by role, export)
src/final_analysis/           dataset validation and classical fits
src/design_density/           simulation of sampling density and placement
results/                      outputs of the scripts above, one folder per analysis
```

The protocol of the simulation study was written before its results were computed:
`results/design_density/PROTOCOL.md`. That protocol refers to its truth parameters as
`results/design_density_truth_fits.json`; the file is `results/design_density/truth_fits.json`.

## Figures

Every figure is written at its printed size (180 mm wide) as a vector PDF and a 600 dpi PNG.

- `results/final_analysis/classical_fits`: kinetic fits, comparison of the kinetic laws, uptake against
  concentration, and the Freundlich exponent under calibration uncertainty.
- `results/design_density/fig1_recovery_vs_density_placement`: how often the generating law is recovered.
- `results/design_density/fig2_identifiability`: how often each parameter is identifiable.
- `results/design_density/fig3_learners_vs_classical`: flexible learners against the selected classical law.

Colours are a three-hue subset of the Okabe-Ito palette with neutral greys for conventional models. Every
series also differs by line style or marker. The main figure was checked under simulated protanopia,
deuteranopia, tritanopia and in greyscale.

## Reproducing the results

Python 3.10 or later. The first three scripts need numpy, scipy, pandas, openpyxl and matplotlib;
the design-density scripts also need scikit-learn and torch.

```
pip install -r requirements.txt
python src/final_analysis/validate_dataset.py
python src/final_analysis/classical_fits.py
python src/final_analysis/make_truth_fits.py
python src/design_density/stage1.py
python src/design_density/stage2.py
python src/design_density/analyze.py
python src/design_density/figures.py
```

Run every command from the repository root. The first three take under two minutes in total.
On the machine used, design-density stage 1 took about 43 min and stage 2 about 79 min
(`results/design_density/stage1.log`, `stage2.log`).

Fitted values reproduce to about four significant figures across platforms. Parameters of the
degenerate laws (Langmuir capacity, Avrami and fractal-PSO capacity) are not identified and
their printed values vary between machines.

## Earlier PINN model

A physics-informed neural network (PINN3D) was developed earlier in this project. It is not part
of this archive or of the manuscript: it fitted the training points closely but generalised poorly
under leave-one-out validation. The design-density study here includes a small kinetic-only PINN as
one of several comparison learners (`src/design_density/ml.py`) and tests that question directly.

## Author

Modelling and analysis code: Tinon Turja Majumder ([@Tinonturja](https://github.com/Tinonturja)).
The experimental data were measured by the co-authors of the accompanying manuscript.

## Citation

Majumder, T. T. (2026). Catkin Biofilm Adsorption: kinetic and isotherm analysis of reactive dye
adsorption on a Saccharum spontaneum (catkin) CMC/PVA/TiO2 biofilm (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.23195585

The DOI above identifies version 1.0.0. The DOI 10.5281/zenodo.23195584 always resolves to the
latest version. Citation metadata is in `CITATION.cff`.

## License

MIT. See `LICENSE`.
