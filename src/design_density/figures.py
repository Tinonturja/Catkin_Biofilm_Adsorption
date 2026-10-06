"""Figures for the design-density study (only those carrying a result). Run after analyze.py.

All three use the shared style in src/figstyle.py and are written as vector PDF and 600 dpi PNG.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'results' / 'design_density'
sys.path.insert(0, str(ROOT / 'src'))
import figstyle as fs  # noqa: E402
fs.apply()
TR = ['T1_PSO', 'T2_PFO', 'T3_Elovich', 'T4_TwoSitePFO', 'T5_PSO+sqrt']
LAB = {'T1_PSO': 'PSO', 'T2_PFO': 'PFO', 'T3_Elovich': 'Elovich', 'T4_TwoSitePFO': 'Two-site PFO', 'T5_PSO+sqrt': r'PSO + $k_{id}\,t^{0.5}$'}
NOISE = 'N3_add_0.03'
sel = pd.read_csv(OUT / 'summary_selection.csv'); par = pd.read_csv(OUT / 'summary_parameters.csv')


def n_axis(a):
    a.set_xscale('log'); a.set_xticks([4, 8, 16, 40]); a.set_xticklabels(['4', '8', '16', '40']); a.minorticks_off()
    a.set_ylim(-0.03, 1.03); a.set_yticks([0, 0.5, 1.0]); a.set_xlim(3.5, 46)


# Figure 1: exact-law and phase-class recovery against the number of observations, by placement
fig, ax = fs.figure(width='double', height_mm=88, nrows=2, ncols=5, sharex=True, sharey=True)
fig.subplots_adjust(left=0.075, right=0.99, top=0.83, bottom=0.135, hspace=0.22, wspace=0.14)
fig.text(0.5325, 0.012, 'Number of observations between 0 and 170 min', ha='center', va='bottom', fontsize=8)
for j, tr in enumerate(TR):
    for r, col in enumerate(['P_exact', 'P_class']):
        a = ax[r, j]; a.axhspan(0.8, 1.03, color=fs.BAND, zorder=0, lw=0)
        for pl, (lab, c, ls, mk) in fs.DESIGN.items():
            d = sel[(sel.truth == tr) & (sel.noise == NOISE) & (sel.placement == pl)].sort_values('n')
            a.plot(d.n, d[col], color=c, ls=ls, marker=mk, ms=3.2, mec='white', mew=0.4, lw=1.1, label=lab)
        A = sel[(sel.truth == tr) & (sel.noise == NOISE) & (sel.placement == 'A_current')][col].iloc[0]
        a.plot([8], [A], marker='*', ms=9, mfc=fs.INK, mec='white', mew=0.5, ls='none', zorder=6, label='design used in this study (n = 8)')
        n_axis(a)
        if r == 0: a.set_title(LAB[tr], pad=4)
ax[0, 0].set_ylabel('P(exact law selected)'); ax[1, 0].set_ylabel('P(correct class selected)')
h, l_ = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc='upper center', ncol=4, bbox_to_anchor=(0.53, 1.0), columnspacing=1.6)
fig.text(0.53, 0.895, 'Law used to generate the synthetic data', ha='center', va='bottom', fontsize=8)
fs.save(fig, OUT / 'fig1_recovery_vs_density_placement')

# Figure 2: practical identifiability of each generating-law parameter against the number of observations
names = {'T1_PSO': [r'$q_e$', r'$k_2$'], 'T2_PFO': [r'$q_e$', r'$k_1$'], 'T3_Elovich': [r'$\alpha$', r'$\beta$'],
         'T4_TwoSitePFO': [r'$q_1$', r'$k_a$', r'$q_2$', r'$k_b$'], 'T5_PSO+sqrt': [r'$q_e$', r'$k_2$', r'$k_{id}$']}
fig, ax = fs.figure(width='double', height_mm=118, nrows=3, ncols=5, sharex=True, sharey=True)
fig.subplots_adjust(left=0.085, right=0.99, top=0.865, bottom=0.10, hspace=0.20, wspace=0.14)
fig.text(0.5375, 0.012, 'Number of observations between 0 and 170 min', ha='center', va='bottom', fontsize=8)
for j, tr in enumerate(TR):
    for r, pl in enumerate(fs.DESIGN):
        a = ax[r, j]; a.axhspan(0.8, 1.03, color=fs.BAND, zorder=0, lw=0)
        for k, nm in enumerate(names[tr]):
            c, mk, ls = fs.PARAM[k]
            d = par[(par.truth == tr) & (par.noise == NOISE) & (par.placement == pl) & (par.param == k)].sort_values('n')
            a.plot(d.n, d.frac_identifiable, color=c, ls=ls, marker=mk, ms=3.2, mec='white', mew=0.4, lw=1.1, label=nm)
        n_axis(a)
        if r == 0:
            a.set_title(LAB[tr], pad=17)
            a.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=4, columnspacing=0.5, handlelength=1.4, handletextpad=0.25, borderaxespad=0.1)
        if j == 0: a.set_ylabel(f'{fs.DESIGN[pl][0].capitalize()}\nfraction identifiable')
fig.text(0.5375, 0.968, 'Law used to generate the synthetic data', ha='center', va='bottom', fontsize=8)
fs.save(fig, OUT / 'fig2_identifiability')
print('stage-1 figures written')

# Figure 3: learners against the AICc-selected classical law (stage 2). One point per design; nothing is clipped.
p2 = OUT / 'summary_stage2.csv'
if p2.exists():
    s2 = pd.read_csv(p2); L = ['M0', 'M4_GPresid', 'M3_NNresid', 'M2s_PINN_8x8', 'M2_PINN_32x32']
    LL = {'M0': 'Generating law, refitted', 'M2_PINN_32x32': 'PINN, 1,283 weights', 'M2s_PINN_8x8': 'PINN, 131 weights',
          'M3_NNresid': 'Neural residual', 'M4_GPresid': 'Gaussian-process residual'}
    D = [('A_current', 8, 'n = 8 (design used)', 'o'), ('B_uniform', 16, 'n = 16, uniform', 's'), ('B_uniform', 40, 'n = 40, uniform', 'D')]
    fig, ax = fs.figure(width='double', height_mm=92, nrows=2, ncols=3, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.185, right=0.955, top=0.84, bottom=0.135, hspace=0.20, wspace=0.08)
    for r, grid in enumerate(['interp', 'extrap']):
        for j, tr in enumerate(['T1_PSO', 'T3_Elovich', 'T4_TwoSitePFO']):
            a = ax[r, j]
            a.axvspan(0, 1.2, color=fs.BAND, zorder=0, lw=0); a.axvline(0, color=fs.INK, lw=0.6)
            for i_, l in enumerate(L):
                for k, (pl, n, lab, mk) in enumerate(D):
                    q_ = s2[(s2.truth == tr) & (s2.placement == pl) & (s2.n == n) & (s2.learner == l) & (s2.grid == grid)]
                    v = q_.median_rel_improvement.iloc[0]; good = bool(q_.adds_information.iloc[0])
                    a.plot(v, i_ + (k - 1) * 0.22, marker=mk, ms=3.8, mfc=fs.SEQ3[k], mec=fs.INK if good else 'white',
                           mew=0.9 if good else 0.4, ls='none', label=lab if (i_ == 0) else None, zorder=3)
            a.set_yticks(range(len(L))); a.set_yticklabels([LL[l] for l in L]); a.tick_params(axis='y', length=0)
            a.set_ylim(-0.6, len(L) - 0.4); a.set_xlim(-9, 1.2); a.set_xticks([-8, -6, -4, -2, 0])
            a.xaxis.grid(True); a.set_axisbelow(True)
            if r == 0: a.set_title(LAB[tr], pad=4)
    for r, lab in enumerate(['Inside 0 to 170 min', 'Extrapolated, 170 to 300 min']):
        ax[r, 2].text(1.04, 0.5, lab, transform=ax[r, 2].transAxes, rotation=270, ha='left', va='center', fontsize=8)
    ax[0, 0].text(0.6, len(L) - 0.5, 'better', rotation=90, ha='center', va='top', fontsize=6.5, color=fs.GREY_DARK)
    h, l_ = ax[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker='o', ms=3.8, mfc='white', mec=fs.INK, mew=0.9, ls='none')); l_.append('black outline: pre-specified gain criterion met')
    fig.legend(h, l_, loc='upper center', ncol=4, bbox_to_anchor=(0.57, 1.0), columnspacing=1.4, handletextpad=0.2)
    fig.text(0.57, 0.905, 'Law used to generate the synthetic data', ha='center', va='bottom', fontsize=8)
    fig.text(0.57, 0.012, 'Median relative gain in RMSE over the AICc-selected classical law', ha='center', va='bottom', fontsize=8)
    fs.save(fig, OUT / 'fig3_learners_vs_classical'); print('stage-2 figure written')
