"""Shared paths and data loading for the final modelling checks.

Everything here is read-only with respect to the dataset. The workbook is never written to.
All scripts in this folder are run from the project root, for example:

    python src/final_analysis/validate_dataset.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'data_of_biofilm.xlsx'
OUT = ROOT / 'results' / 'final_analysis'
OUT.mkdir(parents=True, exist_ok=True)

# Experimental constants that are NOT stored in the workbook.
# Kinetic arm: 100 mL of 40 mg/L dye on 1.2 g of film (V/m = 1/12 L/g, dose 12 g/L). The q column of
#   the workbook is on this scale. VM_KIN enters only the checks that link q to concentration
#   (validate_dataset.py sections 4 and 5, classical_fits.py part C), not the kinetic fits.
# Isotherm arm: V/m = 0.1 L/g, which is recoverable from the sheet itself (Qe / (C0 - Ce)).
C0_KIN = 40.0
VM_KIN = 0.1 / 1.2
VM_ISO = 0.1


def load_kinetics(with_origin=False):
    """Return (t, q) for the catkin kinetic series. The t = 0 row is definitional (q = 0), not a measurement."""
    kin = pd.read_excel(DATA, sheet_name='catkin (pfo pso)')
    t = kin['Time'].to_numpy(float)
    q = kin.iloc[:, 1].to_numpy(float)
    return (t, q) if with_origin else (t[1:], q[1:])


def load_isotherm():
    """Return (C0, Ce, Qe) for the catkin isotherm series (170 min contact, V/m = 0.1 L/g)."""
    iso = pd.read_excel(DATA, sheet_name='catkin')
    return (iso['Initial concentration'].to_numpy(float),
            iso['Ce'].to_numpy(float),
            iso['Qe'].to_numpy(float))


def load_isotherm_absorbance():
    """Return the absorbance readings of the five isotherm samples (the only quantity measured per point)."""
    return pd.read_excel(DATA, sheet_name='catkin')['absorbance'].to_numpy(float)


def load_calibration():
    """Return (concentration mg/L, absorbance) of the six calibration standards.

    The standards are not in the workbook. data/calibration_standards.csv holds them, and an ordinary
    least-squares line through them reproduces the line used in the workbook (A = 0.016684 C - 0.0263).
    """
    c = pd.read_csv(ROOT / 'data' / 'calibration_standards.csv')
    return c.iloc[:, 0].to_numpy(float), c.iloc[:, 1].to_numpy(float)


class Tee:
    """Print to the console and to a log file in results/final_analysis at the same time."""

    def __init__(self, name):
        self.f = open(OUT / name, 'w')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        sys.stdout = self.stdout
        self.f.close()


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
