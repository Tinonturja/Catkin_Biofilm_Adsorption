"""Classical kinetic and isotherm analysis of the catkin dataset.

Part A  Kinetic candidate laws fitted by nonlinear least squares to the 8 measured points (t > 0).
        Reported per law: RMSE, AICc under two parameter-count conventions, BIC, leave-one-out RMSE,
        residual sign runs, and R2 with and without the definitional (0, 0) row.
Part B  Isotherm laws fitted to the 5 measured points, with a small-sample profile interval for the
        Freundlich exponent, and the sensitivity of that exponent to the calibration line.
Part C  A coupled PSO + Freundlich + mass-balance fit at three kinetic V/m values, which shows whether
        the two experiment arms can be described by one equilibrium law.

Run from the project root:  python src/final_analysis/classical_fits.py     (about 1 minute)
Outputs in results/final_analysis/: kinetic_candidates.csv, isotherm_fits.csv, classical_fits.json,
classical_fits.txt, classical_fits.png, isotherm_calibration_sensitivity.csv
"""
import json
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, least_squares

from common import (OUT, Tee, load_kinetics, load_isotherm, load_isotherm_absorbance, load_calibration,
                    rmse, r2, C0_KIN, VM_KIN, VM_ISO)

warnings.filterwarnings('ignore')
log = Tee('classical_fits.txt')
t, y = load_kinetics()
C0, Ce, Qe = load_isotherm()
n_kin, n_iso = len(t), len(Qe)


# ------------------------------------------------------------------------------------------ laws
def pso(t, qe, k2):
    return k2 * qe ** 2 * t / (1 + k2 * qe * t)


def pfo(t, qe, k1):
    return qe * (1 - np.exp(-k1 * t))


def hsdm(t, qe, d):
    """Exact homogeneous diffusion into a sphere (Crank series). d = De / r^2 in 1/min."""
    m = np.arange(1, 200)[:, None]
    return qe * (1 - 6 / np.pi ** 2 * np.sum(np.exp(-m ** 2 * np.pi ** 2 * d * np.atleast_1d(t)) / m ** 2, axis=0))


# name: (function, parameter names, typical scale used to centre the random starts)
KINETIC = {
    'PFO':            (pfo, ['qe', 'k1'], [1, .05]),
    'PSO':            (pso, ['qe', 'k2'], [1, .05]),
    'Elovich':        (lambda t, a, b: np.log1p(a * b * t) / b, ['alpha', 'beta'], [1, 5]),
    'Power law':      (lambda t, a, b: a * t ** b, ['a', 'b'], [.5, .2]),
    'Burst + sqrt(t)': (lambda t, a, b: a + b * np.sqrt(t), ['q_burst', 'k_id'], [.5, .05]),
    'Sphere diffusion (HSDM)': (hsdm, ['qe', 'De/r2'], [1, .001]),
    'Avrami':         (lambda t, a, b, c: a * (1 - np.exp(-(b * t) ** c)), ['qe', 'k', 'n'], [1, .05, 1]),
    'Fractal PSO':    (lambda t, a, b, c: a * b * t ** c / (1 + b * t ** c), ['qe', 'k', 'alpha'], [1, .05, 1]),
    'PSO + k_id sqrt(t)': (lambda t, a, b, c: pso(t, a, b) + c * np.sqrt(t), ['qe', 'k2', 'k_id'], [1, .5, .03]),
    'Two-site PFO':   (lambda t, a, b, c, d: pfo(t, a, b) + pfo(t, c, d), ['q1', 'k_a', 'q2', 'k_b'], [1, .05, 1, .01]),
    'Two-site PSO':   (lambda t, a, b, c, d: pso(t, a, b) + pso(t, c, d), ['q1', 'k_a', 'q2', 'k_b'], [1, .05, 1, .01]),
}


def global_fit(f, scale, tt, yy, starts=150, seed=0):
    """Multi-start least squares in log-parameters (keeps every parameter positive)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        p0 = np.log(scale) + rng.normal(0, 2.0, len(scale))
        try:
            r = least_squares(lambda p: f(tt, *np.exp(p)) - yy, p0, max_nfev=4000)
        except Exception:
            continue
        if np.all(np.isfinite(r.fun)) and (best is None or r.cost < best.cost):
            best = r
    return best


def aicc(sse, n, K):
    """Least-squares AICc. K is the number of estimated quantities."""
    return n * np.log(sse / n) + 2 * K + (2 * K * (K + 1) / (n - K - 1) if n - K - 1 > 0 else np.inf)


# ------------------------------------------------------------------------------------------ Part A
rows, curves = [], {}
for name, (f, pnames, scale) in KINETIC.items():
    k = len(pnames)
    b = global_fit(f, scale, t, y)
    p = np.exp(b.x)
    fit = f(t, *p)
    res = fit - y
    sse = float((res ** 2).sum())
    loo = []
    for j in range(n_kin):
        bj = global_fit(f, scale, np.delete(t, j), np.delete(y, j), starts=50, seed=j + 1)
        loo.append(f(t[j:j + 1], *np.exp(bj.x))[0] - y[j])
    rows.append({
        'model': name, 'k': k,
        'parameters': '; '.join(f'{a} = {v:.4g}' for a, v in zip(pnames, p)),
        'RMSE': np.sqrt(sse / n_kin),
        'AICc_K=k': aicc(sse, n_kin, k),            # noise variance not counted
        'AICc_K=k+1': aicc(sse, n_kin, k + 1),      # noise variance counted as an estimated parameter
        'BIC': n_kin * np.log(sse / n_kin) + (k + 1) * np.log(n_kin),
        'LOO_RMSE': float(np.sqrt(np.mean(np.square(loo)))),
        'sign_runs': int(1 + (np.sign(res[1:]) != np.sign(res[:-1])).sum()),
        'R2_measured_points': r2(y, fit),
        'R2_with_origin_row': r2(np.r_[0, y], np.r_[0 if name != 'Burst + sqrt(t)' else p[0], fit]),
        'q_at_300min_extrapolated': float(f(np.array([300.0]), *p)[0]),
    })
    curves[name] = (f, p)

K = pd.DataFrame(rows)
for c in ('AICc_K=k', 'AICc_K=k+1', 'BIC'):
    K['d' + c] = K[c] - K[c].min()
w = np.exp(-0.5 * K['dAICc_K=k+1'])
K['akaike_weight'] = w / w.sum()
K = K.sort_values('AICc_K=k+1').reset_index(drop=True)
K.to_csv(OUT / 'kinetic_candidates.csv', index=False)

print(f'[A] Kinetic candidate laws, n = {n_kin} measured points (t > 0), sorted by AICc with K = k + 1')
show = K[['model', 'k', 'RMSE', 'dAICc_K=k', 'dAICc_K=k+1', 'akaike_weight', 'dBIC', 'LOO_RMSE', 'sign_runs',
          'R2_measured_points', 'R2_with_origin_row']]
print(show.round(4).to_string(index=False))
print('\n    Fitted parameters')
for _, r_ in K.iterrows():
    print(f"    {r_['model']:<26s} {r_['parameters']}")
unbounded = [r_['model'] for _, r_ in K.iterrows() if curves[r_['model']][1][0] > 100 * y.max()]
print('\n    Notes')
print('    - AICc with K = k + 1 counts the residual variance as an estimated quantity; K = k does not. Both are given.')
print('    - R2_with_origin_row adds the (0, 0) row, which is fixed by definition and not a measurement.')
print('    - Burst + sqrt(t) has q(0) = q_burst, not 0, so its R2 with the origin row is not comparable with the others.')
print('    - Akaike weights use the K = k + 1 column.')
if unbounded:
    print('    - Capacity parameter unbounded (fit not identified): ' + ', '.join(unbounded) + '.')

# ------------------------------------------------------------------------------------------ Part B
ISO = {
    'Freundlich': (lambda ce, a, b: a * ce ** b, ['Kf', 'n'], [.05, 1]),
    'Langmuir':   (lambda ce, a, b: a * b * ce / (1 + b * ce), ['qmax', 'KL'], [10, .01]),
    'Linear':     (lambda ce, a: a * ce, ['K'], [.07]),
}


def q_from_c0(g, c0, vm):
    """Solve q = g(C0 - q/vm) for q, so the prediction uses C0 only (Ce is itself a model output)."""
    return np.array([brentq(lambda q: q - g(max(c - q / vm, 1e-12)), 0, c * vm * (1 - 1e-9)) for c in np.atleast_1d(c0)])


irows = []
for name, (f, pnames, scale) in ISO.items():
    k = len(pnames)
    b = global_fit(f, scale, Ce, Qe, starts=200)
    p = np.exp(b.x)
    sse = float(((f(Ce, *p) - Qe) ** 2).sum())
    loo = []
    for j in range(n_iso):
        bj = global_fit(f, scale, np.delete(Ce, j), np.delete(Qe, j), starts=60, seed=j + 1)
        loo.append(f(Ce[j:j + 1], *np.exp(bj.x))[0] - Qe[j])
    irows.append({
        'model': name, 'k': k,
        'parameters': '; '.join(f'{a} = {v:.4g}' for a, v in zip(pnames, p)),
        'RMSE_vs_Ce': np.sqrt(sse / n_iso),
        'RMSE_predicted_from_C0': rmse(Qe, q_from_c0(lambda c: f(c, *p), C0, VM_ISO)),
        'AICc_K=k': aicc(sse, n_iso, k),
        'AICc_K=k+1': aicc(sse, n_iso, k + 1),
        'LOO_RMSE_vs_Ce': float(np.sqrt(np.mean(np.square(loo)))),
        'R2': r2(Qe, f(Ce, *p)),
    })
    curves[name] = (f, p)
I = pd.DataFrame(irows)
I['dAICc_K=k'] = I['AICc_K=k'] - I['AICc_K=k'].min()
I.to_csv(OUT / 'isotherm_fits.csv', index=False)
print(f'\n[B] Isotherm laws, n = {n_iso} measured points')
print(I[['model', 'k', 'parameters', 'RMSE_vs_Ce', 'RMSE_predicted_from_C0', 'dAICc_K=k', 'LOO_RMSE_vs_Ce', 'R2']].round(4).to_string(index=False))
print(f'    dAICc is given for K = k only: with n = {n_iso}, counting the residual variance leaves too few degrees of freedom.')

# profile interval for the Freundlich exponent
fF = ISO['Freundlich'][0]
kf_hat, n_hat = curves['Freundlich'][1]
sse0 = float(((fF(Ce, kf_hat, n_hat) - Qe) ** 2).sum())
grid = np.linspace(0.6, 2.2, 3201)


def sse_at_n(ce, qe, nv):
    kf = (qe * ce ** nv).sum() / (ce ** (2 * nv)).sum()       # Kf is linear given n, so its optimum is closed-form
    return float(((kf * ce ** nv - qe) ** 2).sum())


ratio = np.array([sse_at_n(Ce, Qe, nv) for nv in grid]) / sse0
dof = n_iso - 2
thr_F = 1 + stats.f.ppf(0.95, 1, dof) / dof                  # small-sample threshold, residual variance estimated
thr_chi2 = np.exp(3.84 / n_iso)                              # large-sample threshold, shown for comparison only
ci_n = [float(grid[ratio <= thr_F].min()), float(grid[ratio <= thr_F].max())]
ci_n_chi2 = [float(grid[ratio <= thr_chi2].min()), float(grid[ratio <= thr_chi2].max())]
print(f'    Freundlich exponent n = {n_hat:.3f}. 95 % profile interval at the workbook calibration:')
print(f'      small-sample F threshold ({dof} residual degrees of freedom): {ci_n[0]:.2f} to {ci_n[1]:.2f}')
print(f'      large-sample chi-square threshold (too narrow for n = {n_iso}): {ci_n_chi2[0]:.2f} to {ci_n_chi2[1]:.2f}')

# sensitivity of the exponent to the calibration line. Each isotherm point is one absorbance reading; Ce and Qe
# are both derived from it through the same line, so a calibration error moves all five points together.
A_iso = load_isotherm_absorbance()
xc, yc = load_calibration()


def cal_line(x, y_, through_origin):
    if through_origin:
        return (x * y_).sum() / (x * x).sum(), 0.0
    a_, b_ = np.polyfit(x, y_, 1)
    return a_, b_


def freundlich_from_line(a_, b_, absorb):
    ce = (absorb - b_) / a_
    qe = (C0 - ce) * VM_ISO
    if a_ <= 0 or np.any(ce <= 0) or np.any(qe <= 0):
        return None
    r_ = least_squares(lambda p_: np.exp(p_[0]) * ce ** p_[1] - qe, [np.log(.03), 1.2])
    return float(np.exp(r_.x[0])), float(r_.x[1]), ce


cal_rows = []
for label, x, y_, origin in [
        ('with intercept, 6 standards (line used in the workbook)', xc, yc, False),
        ('through origin, 6 standards', xc, yc, True),
        ('with intercept, highest standard dropped', xc[:-1], yc[:-1], False),
        ('through origin, highest standard dropped', xc[:-1], yc[:-1], True)]:
    a_, b_ = cal_line(x, y_, origin)
    Kf_, n_, ce_ = freundlich_from_line(a_, b_, A_iso)
    cal_rows.append({'calibration': label, 'slope': a_, 'intercept': b_, 'Kf': Kf_, 'n': n_,
                     'removal_%_at_20': 100 * (1 - ce_[0] / C0[0]), 'removal_%_at_60': 100 * (1 - ce_[-1] / C0[-1])})
CAL = pd.DataFrame(cal_rows)
CAL.to_csv(OUT / 'isotherm_calibration_sensitivity.csv', index=False)
a_w, b_w = cal_line(xc, yc, False)
res_c = yc - (a_w * xc + b_w)
s_c = float(np.sqrt((res_c ** 2).sum() / (len(xc) - 2)))
Xd = np.c_[xc, np.ones(len(xc))]
cov = s_c ** 2 * np.linalg.inv(Xd.T @ Xd)
rng = np.random.default_rng(20261006)
n_draws = []
for _ in range(4000):
    z = rng.multivariate_normal([0, 0], cov) * np.sqrt((len(xc) - 2) / rng.chisquare(len(xc) - 2))
    out_ = freundlich_from_line(a_w + z[0], b_w + z[1], A_iso)
    if out_ is not None:
        n_draws.append(out_[1])
n_draws = np.array(n_draws)
mc_n = [float(np.percentile(n_draws, 2.5)), float(np.median(n_draws)), float(np.percentile(n_draws, 97.5))]
p_gt1 = float(np.mean(n_draws > 1))
print('\n[B2] Sensitivity of the Freundlich exponent to the calibration line')
print(CAL.round(4).to_string(index=False))
print(f'    Calibration residual sd {s_c:.3f} absorbance ({s_c / a_w:.1f} mg/L) from {len(xc)} standards. Monte Carlo over the')
print(f'    line ({len(n_draws)} valid draws of 4000): n median {mc_n[1]:.2f}, 95 % range {mc_n[0]:.2f} to {mc_n[2]:.2f}, P(n > 1) = {p_gt1:.2f}.')
lang = curves['Langmuir'][1]
if lang[0] > 100 * Qe.max():
    print(f'    Langmuir capacity is unbounded (qmax = {lang[0]:.3g}, KL = {lang[1]:.3g}); its fit equals the linear law.')

# ------------------------------------------------------------------------------------------ Part C
def q_eq(c0, vm, Kf, nn):
    try:
        return brentq(lambda q: q - Kf * max(c0 - q / vm, 1e-12) ** nn, 0, c0 * vm * (1 - 1e-9))
    except ValueError:
        return np.nan


def coupled_residual(lp, vm, fit_vm):
    k2, Kf, nn = np.exp(lp[:3])
    vmk = np.exp(lp[3]) if fit_vm else vm
    qe = q_eq(C0_KIN, vmk, Kf, nn)
    return np.r_[pso(t, qe, k2) - y, np.array([q_eq(c, VM_ISO, Kf, nn) for c in C0]) - Qe]


print('\n[C] One PSO rate constant + one Freundlich law + mass balance, fitted to both arms together')
crows = []
for label, vm, fit_vm in [('1/12 L/g (100 mL on 1.2 g)', VM_KIN, False), ('1/24 L/g (50 mL on 1.2 g)', 1 / 24, False), ('fitted', 0.05, True)]:
    best = None
    for s in range(80):
        p0 = np.r_[np.log([.1, .03, 1.2]) + np.random.default_rng(s).normal(0, .7, 3), np.log(0.05)][:4 if fit_vm else 3]
        try:
            r_ = least_squares(coupled_residual, p0, args=(vm, fit_vm), max_nfev=3000)
        except Exception:
            continue
        if np.all(np.isfinite(r_.fun)) and (best is None or r_.cost < best.cost):
            best = r_
    p = np.exp(best.x)
    crows.append({'kinetic_VM': label, 'k2': p[0], 'Kf': p[1], 'n': p[2], 'VM_fitted': p[3] if fit_vm else np.nan,
                  'RMSE_kinetic': rmse(0, best.fun[:n_kin]), 'RMSE_isotherm': rmse(0, best.fun[n_kin:])})
Cdf = pd.DataFrame(crows)
print(Cdf.round(4).to_string(index=False))
print('    Separate fits for reference: PSO RMSE %.4f, Freundlich RMSE (from C0) %.4f.' % (
    K.loc[K.model == 'PSO', 'RMSE'].iloc[0], I.loc[I.model == 'Freundlich', 'RMSE_predicted_from_C0'].iloc[0]))
print('    The kinetic q column is taken as given; only the V/m used in the mass balance is varied.')
print(f"    V/m when fitted as a free parameter: {Cdf.loc[Cdf.kinetic_VM == 'fitted', 'VM_fitted'].iloc[0]:.4f} L/g.")

# ------------------------------------------------------------------------------------------ outputs
json.dump({
    'kinetics': K.to_dict(orient='records'),
    'isotherm': I.to_dict(orient='records'),
    'freundlich_n_profile_CI95_F': ci_n,
    'freundlich_n_profile_CI95_chi2_large_sample': ci_n_chi2,
    'isotherm_calibration_sensitivity': CAL.to_dict(orient='records'),
    'freundlich_n_calibration_MC_2.5_50_97.5': mc_n,
    'freundlich_n_calibration_MC_P_gt_1': p_gt1,
    'coupled': Cdf.to_dict(orient='records'),
    'n_kinetic_points': n_kin, 'n_isotherm_points': n_iso,
}, open(OUT / 'classical_fits.json', 'w'), indent=1, default=float)

# ------------------------------------------------------------------------------------------ figure
sys.path.insert(0, str(OUT.parents[1] / 'src'))
import figstyle as fs  # noqa: E402

fs.apply()
fig, axs = fs.figure(width='double', height_mm=140, nrows=2, ncols=2)
fig.subplots_adjust(left=0.065, right=0.985, top=0.945, bottom=0.075, hspace=0.42, wspace=0.26)
(axA, axB), (axC, axD) = axs
pos = axB.get_position()                      # room for the law names on the left of panel b
axB.set_position([pos.x0 + 0.115, pos.y0, pos.width - 0.115, pos.height])

# (a) kinetic fits with the residuals of the conventional and the best law underneath
tt = np.linspace(0.01, 178, 600)
for name in ['PFO', 'PSO', 'Elovich', 'Burst + sqrt(t)']:
    f, p = curves[name]
    c, ls, lw = fs.LAW[name]
    lab = {'PFO': 'Pseudo-first-order', 'PSO': 'Pseudo-second-order', 'Burst + sqrt(t)': r'Burst + $t^{0.5}$'}.get(name, name)
    axA.plot(tt, f(tt, *p), color=c, ls=ls, lw=lw, label=lab)
axA.plot(t, y, label='Measured', **fs.MEASURED)
axA.set_xlim(0, 180)
axA.set_ylim(0, 1.3)
axA.set_xticks(range(0, 181, 30))
axA.set_xlabel('Contact time (min)')
axA.set_ylabel(r'$q_t$ (mg g$^{-1}$)')
h_, l_ = axA.get_legend_handles_labels()
axA.legend(h_[::-1], l_[::-1], loc='lower right', bbox_to_anchor=(1.0, 0.02))
fs.panel(axA, 'a', 'Kinetics, 40 mg L$^{-1}$')

# (b) all candidate laws: in-sample and leave-one-out error, ordered by leave-one-out error
show_b = K[~K.model.isin(['Avrami', 'Fractal PSO'])].sort_values('LOO_RMSE', ascending=False).reset_index(drop=True)
nice = {'PFO': 'Pseudo-first-order', 'PSO': 'Pseudo-second-order', 'Burst + sqrt(t)': r'Burst + $t^{0.5}$',
        'PSO + k_id sqrt(t)': r'PSO + $k_{id}\,t^{0.5}$', 'Sphere diffusion (HSDM)': 'Sphere diffusion',
        'Two-site PFO': 'Two-site first-order', 'Two-site PSO': 'Two-site second-order'}
for i_, r_ in show_b.iterrows():
    top_ = r_['model'] == 'Burst + sqrt(t)'
    col = fs.BLUE if top_ else fs.GREY_DARK
    axB.plot([r_['RMSE'], r_['LOO_RMSE']], [i_, i_], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
    axB.plot(r_['RMSE'], i_, marker='o', ms=4.4, mfc='white', mec=col, mew=1.0, ls='none', zorder=3)
    axB.plot(r_['LOO_RMSE'], i_, marker='o', ms=4.4, mfc=col, mec=col, ls='none', zorder=3)
    axB.text(0.198, i_, f"{r_['akaike_weight']:.2f}" if r_['akaike_weight'] >= 0.005 else '<0.01', ha='right', va='center', fontsize=7)
axB.set_yticks(range(len(show_b)))
axB.set_yticklabels([f"{nice.get(m, m)} ({k_})" for m, k_ in zip(show_b.model, show_b.k)])
axB.set_xlim(0, 0.20)
axB.set_xticks([0, 0.05, 0.10, 0.15])
axB.set_xticklabels(['0', '0.05', '0.10', '0.15'])
axB.set_ylim(-0.6, len(show_b) - 0.4)
axB.set_xlabel(r'Root-mean-square error (mg g$^{-1}$)')
axB.xaxis.grid(True)
axB.set_axisbelow(True)
axB.tick_params(axis='y', length=0)
axB.text(0.198, len(show_b) - 0.45, 'Akaike\nweight', ha='right', va='bottom', fontsize=7)
axB.plot([], [], marker='o', ms=4.4, mfc='white', mec=fs.GREY_DARK, mew=1.0, ls='none', label='Fit to all 8 points')
axB.plot([], [], marker='o', ms=4.4, mfc=fs.GREY_DARK, mec=fs.GREY_DARK, ls='none', label='Leave-one-out')
axB.legend(loc='center', bbox_to_anchor=(0.56, 0.74), handletextpad=0.3)
fs.panel(axB, 'b', 'Kinetic laws compared (number of parameters)', x=-0.50)

# (c) uptake at 170 min against residual concentration
cc = np.linspace(0, 38, 200)
for name in ['Linear', 'Freundlich']:
    f, p = curves[name]
    c, ls, lw = fs.LAW[name]
    axC.plot(cc, f(np.maximum(cc, 1e-9), *p), color=c, ls=ls, lw=lw, label=name)
axC.plot(Ce, Qe, label='Measured', **fs.MEASURED)
axC.set_xlim(0, 40)
axC.set_ylim(0, 2.8)
axC.set_xlabel(r'$C_e$ (mg L$^{-1}$)')
axC.set_ylabel(r'$q_e$ at 170 min (mg g$^{-1}$)')
h_, l_ = axC.get_legend_handles_labels()
axC.legend(h_[::-1], l_[::-1], loc='upper left')
fs.panel(axC, 'c', 'Uptake against concentration')

# (d) how far the calibration line moves the Freundlich exponent
axD.hist(n_draws[(n_draws > 0.3) & (n_draws < 2.7)], bins=48, color=fs.GREY_LIGHT, edgecolor='white', linewidth=0.3)
top = axD.get_ylim()[1]
axD.set_ylim(0, top * 1.50)
axD.axvline(1.0, color=fs.INK, lw=0.8, ls=(0, (1, 1.5)))
axD.text(0.97, top * 1.47, 'n = 1 (linear)', ha='right', va='top', fontsize=7)
axD.plot(ci_n, [top * 1.10] * 2, color=fs.BLUE, lw=1.6, solid_capstyle='butt')
axD.plot([n_hat], [top * 1.10], marker='o', ms=4.4, color=fs.BLUE, ls='none')
axD.text(ci_n[1] + 0.05, top * 1.10, 'workbook calibration,\n95 % profile interval', ha='left', va='center', fontsize=7)
for lab, row, side in [('through origin', CAL.iloc[1], -1), ('highest standard dropped', CAL.iloc[2], 1)]:
    axD.plot([row['n']], [top * 1.27], marker='v', ms=5, mfc='white', mec=fs.INK, mew=0.9, ls='none')
    axD.text(row['n'] + side * 0.06, top * 1.27, lab, ha='left' if side > 0 else 'right', va='center', fontsize=7)
axD.set_xlim(0.3, 2.7)
axD.set_yticks([])
axD.spines['left'].set_visible(False)
axD.set_xlabel('Freundlich exponent $n$')
axD.set_ylabel('Draws of the calibration line')
fs.panel(axD, 'd', 'Exponent under calibration uncertainty')
fs.save(fig, OUT / 'classical_fits')
print('\nSaved kinetic_candidates.csv, isotherm_fits.csv, isotherm_calibration_sensitivity.csv, classical_fits.json, classical_fits.png, classical_fits.pdf')
log.close()
