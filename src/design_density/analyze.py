"""Pre-registered summaries and decision rules for the design-density study (PROTOCOL.md sections 8-9).
Run from the project root after stage 1 and stage 2: python src/design_density/analyze.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'results' / 'design_density'
S1 = pd.read_csv(OUT / 'stage1_replicates.csv.gz'); LOO = pd.read_csv(OUT / 'stage1_loo_stability.csv.gz'); PR = pd.read_csv(OUT / 'stage1_profile_coverage.csv.gz')
S2p = OUT / 'stage2_replicates.csv.gz'; S2 = pd.read_csv(S2p) if S2p.exists() else None
R = {}
def region(p): return 'reliable' if p >= 0.8 else ('transition' if p >= 0.5 else 'unreliable')
def entropy(s):
    c = s.value_counts(normalize=True).to_numpy(); return float(-(c * np.log2(c)).sum())

# ---------------- selection probabilities ----------------
g = S1.groupby(['truth', 'noise', 'placement', 'n'])
sel = g.agg(P_exact=('exact', 'mean'), P_class=('cls', 'mean'), w_truth=('w_truth', 'mean')).reset_index()
sel['entropy_bits'] = g['selected'].apply(entropy).values
sel['region_exact'] = sel.P_exact.map(region); sel['region_class'] = sel.P_class.map(region)
sel.to_csv(OUT / 'summary_selection.csv', index=False)
conf = S1[(S1.placement == 'A_current') & (S1.noise == 'N3_add_0.03')].groupby('truth')['selected'].value_counts(normalize=True).unstack(fill_value=0).round(3)
conf.to_csv(OUT / 'confusion_A_N3.csv'); R['confusion_A_N3'] = conf.to_dict(orient='index')

# ---------------- parameter recovery / identifiability ----------------
rows = []
for key, d in g:
    for j in range(4):
        c = f'p{j}_logerr'
        if c not in d or d[c].isna().all(): continue
        e = d[c].dropna()
        rows.append(dict(truth=key[0], noise=key[1], placement=key[2], n=key[3], param=j, n_fits=len(e), median_abs_logerr=float(e.abs().median()),
                         rmse_logerr=float(np.sqrt((e ** 2).mean())), bias_logerr=float(e.median()), coverage=float(d[f'p{j}_cover'].dropna().mean()),
                         frac_identifiable=float(d[f'p{j}_ident'].dropna().mean())))
par = pd.DataFrame(rows); par['identifiable_condition'] = par.frac_identifiable >= 0.8; par.to_csv(OUT / 'summary_parameters.csv', index=False)

# ---------------- prediction (M0 oracle, M1 selected) ----------------
pred = g[[c for c in S1.columns if c[:3] in ('M0_', 'M1_')]].median().reset_index(); pred.to_csv(OUT / 'summary_prediction_median_rmse.csv', index=False)

# ---------------- hypotheses ----------------
N3 = sel[sel.noise == 'N3_add_0.03']
A = N3[N3.placement == 'A_current'].set_index('truth')
R['H1'] = dict(P_exact_at_A=A.P_exact.round(3).to_dict(), P_class_at_A=A.P_class.round(3).to_dict(),
               falsified=bool((A.P_exact >= 0.8).sum() >= 4))
B = N3[N3.placement == 'B_uniform'].groupby('n').P_exact.mean()
R['H2'] = dict(mean_P_exact_by_n_B=B.round(3).to_dict(), diff_40_minus_8=float(B[40] - B[8]), falsified=bool(B[40] - B[8] < 0.10))
pl = N3[N3.placement != 'A_current'].groupby(['n', 'placement']).P_exact.mean().unstack()
spread = (pl.max(axis=1) - pl.min(axis=1))
R['H3'] = dict(mean_P_exact_by_n_and_placement=pl.round(3).to_dict(orient='index'), spread=spread.round(3).to_dict(),
               falsified=bool(spread[8] < 0.10 and spread[16] < 0.10))
R['H7'] = dict(P_select_PSOsqrt_at_A_N3={t: float(conf.loc[t].get('PSO+sqrt', 0.0)) for t in conf.index},
               compatible_truths=[t for t in conf.index if conf.loc[t].get('PSO+sqrt', 0.0) >= 0.05])
lo = LOO.groupby(['truth', 'noise']).agg(mean_agree=('agree_frac', 'mean'), frac_agree_le_1of8=('agree_frac', lambda s: float((s <= 0.125 + 1e-9).mean())),
                                         median_agree=('agree_frac', 'median')).reset_index()
lo.to_csv(OUT / 'summary_loo_stability.csv', index=False); R['H7_loo'] = lo.round(3).to_dict(orient='records')
R['profile_coverage_PSO'] = PR.groupby('n')[['p0_cover', 'p1_cover', 'p0_closed', 'p1_closed', 'p0_logwidth', 'p1_logwidth']].mean().round(3).to_dict(orient='index')

# ---------------- stage 2 ----------------
if S2 is not None:
    learners = ['M0', 'M2_PINN_32x32', 'M2s_PINN_8x8', 'M3_NNresid', 'M4_GPresid']; out = []
    for (tr, pl_, n), d in S2.groupby(['truth', 'placement', 'n']):
        for L in learners:
            for grid in ['interp', 'early', 'late', 'extrap']:
                a, b = d[f'{L}_{grid}'].to_numpy(), d[f'M1_{grid}'].to_numpy()
                rel = np.median((b - a) / b)
                try: pval = float(wilcoxon(a, b, alternative='less').pvalue)
                except ValueError: pval = 1.0
                out.append(dict(truth=tr, placement=pl_, n=n, learner=L, grid=grid, n_reps=len(d), median_rmse=float(np.median(a)), median_rmse_M1=float(np.median(b)),
                                median_rel_improvement=float(rel), p_one_sided=pval, adds_information=bool(pval < 0.05 and rel >= 0.10)))
    s2 = pd.DataFrame(out); s2.to_csv(OUT / 'summary_stage2.csv', index=False)
    A2 = s2[(s2.placement == 'A_current') & (s2.grid == 'interp') & (s2.learner.isin(['M3_NNresid', 'M4_GPresid']))]
    R['H4'] = dict(rows=A2[['truth', 'learner', 'median_rel_improvement', 'p_one_sided', 'adds_information']].round(4).to_dict(orient='records'),
                   falsified=bool(A2.adds_information.any()))
    t1 = S2[S2.truth == 'T1_PSO']; rec = {}
    for (pl_, n), d in t1.groupby(['placement', 'n']):
        for tag, k, q in [('PINN_32x32', 'PINN_k2_logerr', 'PINN_qe_logerr'), ('PINN_8x8', 'PINNs_k2_logerr', 'PINNs_qe_logerr')]:
            ok = (d[k].abs() < np.log(1.25)) & (d[q].abs() < np.log(1.10))
            rec[f'{pl_}|{n}|{tag}'] = dict(frac_recovered=float(ok.mean()), median_abs_k2_logerr=float(d[k].abs().median()), median_abs_qe_logerr=float(d[q].abs().median()), recovers=bool(ok.mean() >= 0.8))
    R['H5_H6_PINN_recovery_T1'] = rec
    R['H5'] = dict(falsified=bool(rec['A_current|8|PINN_32x32']['recovers']))
    B40 = s2[(s2.placement == 'B_uniform') & (s2.n == 40) & (s2.learner.isin(['M2_PINN_32x32', 'M2s_PINN_8x8']))]
    R['H6'] = dict(rows=B40[['truth', 'learner', 'grid', 'median_rel_improvement', 'p_one_sided', 'adds_information']].round(4).to_dict(orient='records'),
                   supported=bool(B40.adds_information.any() or rec['B_uniform|40|PINN_32x32']['recovers'] or rec['B_uniform|40|PINN_8x8']['recovers']))
json.dump(R, open(OUT / 'hypotheses.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ('confusion_A_N3', 'H7_loo', 'H6', 'H4')}, indent=1, default=float)[:6000])
