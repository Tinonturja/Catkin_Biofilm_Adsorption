"""Dataset validation for data/data_of_biofilm.xlsx. Read-only: nothing in the workbook is changed.

Checks, in order:
  1. the calibration line implied by the isotherm sheet
  2. every derived column in the isotherm sheet against a recomputation
  3. the kinetic series (monotonicity, increments, rates)
  4. consistency of the kinetic q(t) series with the 'removal% against time' sheet
  5. consistency of the 170-min kinetic point with the isotherm
  6. the size and shape of the experimental design

Run from the project root:  python src/final_analysis/validate_dataset.py
Output: results/final_analysis/dataset_validation.txt
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common import DATA, Tee, C0_KIN, VM_KIN

log = Tee('dataset_validation.txt')
X = pd.ExcelFile(DATA)
print('sheets:', X.sheet_names)

iso = pd.read_excel(X, 'catkin')
kin = pd.read_excel(X, 'catkin (pfo pso)')
rem_conc = pd.read_excel(X, 'removal against % concentration')
rem_time = pd.read_excel(X, 'removal% against time', header=None).iloc[2:, :2].astype(float).reset_index(drop=True)
rem_time.columns = ['t', 'catkin']

for name, d in [('catkin', iso), ('catkin (pfo pso)', kin)]:
    numeric = all(np.issubdtype(t, np.number) for t in d.dtypes)
    print(f'{name}: shape {d.shape}, missing values {int(d.isna().sum().sum())}, all numeric {numeric}')

# ---------------------------------------------------------------- 1. calibration
A = iso['absorbance'].to_numpy()
Ce = iso['Ce'].to_numpy()
C0 = iso['Initial concentration'].to_numpy(float)
Qe = iso['Qe'].to_numpy()
slope, icpt = np.polyfit(Ce, A, 1)
print('\n[1] Calibration implied by the catkin isotherm sheet')
print(f'    A = {slope:.6f} * C(mg/L) {icpt:+.5f}; largest residual {abs(A - (slope * Ce + icpt)).max():.2e}')

# ---------------------------------------------------------------- 2. isotherm chain
recomputed = {
    'Co-Ce': C0 - Ce,
    'Qe': (C0 - Ce) * 0.1,
    'LN CE': np.log(Ce),
    'LN QE': np.log(Qe),
    'Ce/Qe': Ce / Qe,
    'removal percentage(catkin)': 100 * (C0 - Ce) / C0,
}
print('\n[2] Isotherm derived columns, largest |sheet - recomputed|')
for col, val in recomputed.items():
    print(f'    {col:<28s} {abs(iso[col].to_numpy() - val).max():.2e}')
print('    V/m implied by Qe/(C0-Ce):', np.round(Qe / (C0 - Ce), 6).tolist())
print('    removal sheet equals isotherm-sheet removal:',
      bool(np.allclose(rem_conc['removal percentage(catkin)'], iso['removal percentage(catkin)'], atol=1e-5)))

# ---------------------------------------------------------------- 3. kinetics
t = kin['Time'].to_numpy(float)
q = kin.iloc[:, 1].to_numpy()
print('\n[3] Kinetic series')
print('    t (min):', t.astype(int).tolist())
print('    q (mg/g):', q.tolist())
print('    strictly increasing:', bool((np.diff(q) > 0).all()))
print('    increments (mg/g):', np.round(np.diff(q), 4).tolist())
print('    interval rates (mg/g/min):', np.round(np.diff(q) / np.diff(t), 4).tolist())
print(f'    q(10)/q(170) = {q[1] / q[-1]:.3f}; gain over the last interval = {100 * (q[-1] - q[-2]) / q[-1]:.2f} % of q(170)')

# ---------------------------------------------------------------- 4. q(t) against the removal sheet
Rk = rem_time['catkin'].to_numpy()
print('\n[4] V/m implied by q / (C0 * removal/100), C0 = 40 mg/L')
ratio = q[1:] / (C0_KIN * Rk / 100)
print('    catkin:', np.round(ratio, 4).tolist(), f' range {ratio.min():.4f} to {ratio.max():.4f}')
print('    A constant ratio would mean the q column and the removal column agree.')
for vm in (VM_KIN, 1 / 24):
    print(f'    catkin removal % implied by q at V/m = {vm:.4f}:', np.round(100 * q[1:] / (C0_KIN * vm), 1).tolist())
print('    catkin removal % stored in the sheet:        ', np.round(Rk, 1).tolist())
print('    The q(t) series is the reference in every fit; the catkin removal-vs-time column is not used.')

# ---------------------------------------------------------------- 5. kinetic end point against the isotherm
print('\n[5] 170-min kinetic point against the isotherm (isotherm q interpolated at the same Ce)')
for vm in (VM_KIN, 0.0543, 1 / 24, 0.1):
    ce = C0_KIN - q[-1] / vm
    inside = Ce.min() <= ce <= Ce.max()
    qi = f'{np.interp(ce, Ce, Qe):.3f}' if inside else f'outside the isotherm Ce range ({Ce.min():.1f} to {Ce.max():.1f})'
    print(f'    V/m = {vm:.4f}: Ce = {ce:.2f} mg/L, kinetic q = {q[-1]:.3f}, isotherm q = {qi}')
vm_star = brentq(lambda vm: np.interp(C0_KIN - q[-1] / vm, Ce, Qe) - q[-1], 0.045, 0.099)
print(f'    V/m that puts the 170-min kinetic point on the isotherm: {vm_star:.4f} L/g')

# ---------------------------------------------------------------- 6. design
print(f'\n[6] Design: {len(q) - 1} kinetic measurements (one C0, one V/m) + {len(Qe)} isotherm measurements '
      f'(one contact time, one V/m) = {len(q) - 1 + len(Qe)} measurements.')
print('    V/m takes one value per arm, so dosage is confounded with the experiment arm and no dosage effect can be learned.')
log.close()
