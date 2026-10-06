"""Stage 1 (classical) of the design-density study. Implements PROTOCOL.md sections 2-6.
Run from the project root: python src/design_density/stage1.py"""
import sys, json, time, zlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import MODELS, SINGLE_PHASE, fit, aicc, wald_logse, design, add_noise, T_MAX, T_EXTRAP
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'results' / 'design_density'; OUT.mkdir(parents=True, exist_ok=True)
TF = json.load(open(OUT / 'truth_fits.json'))
TRUTHS = {'T1_PSO': 'PSO', 'T2_PFO': 'PFO', 'T3_Elovich': 'Elovich', 'T4_TwoSitePFO': 'TwoSitePFO', 'T5_PSO+sqrt': 'PSO+sqrt'}
DENS = [4, 6, 8, 12, 16, 24, 40]
GRIDS = {'interp': np.linspace(0, T_MAX, 171), 'early': np.linspace(0.5, 20, 40), 'late': np.linspace(120, T_MAX, 51), 'extrap': np.linspace(T_MAX + 1, T_EXTRAP, 130)}
REPS = 500

def seed(label): return zlib.crc32(label.encode()) ^ 20260929

def draw_truth(rng, law):
    lp = np.array(TF[law]['logp']); s = np.minimum(np.nan_to_num(np.array(TF[law]['logse']), nan=0.5), 0.5)
    f = MODELS[law][0]
    for _ in range(100):
        p = np.exp(lp + rng.normal(0, s))
        if 0.8 <= f(np.array([T_MAX]), *p)[0] <= 1.6: return p
    return np.exp(lp)

def one_replicate(rng, law, noise, t):
    f = MODELS[law][0]; p = draw_truth(rng, law); y = add_noise(rng, f(t, *p), noise); n = len(t)
    fits = {}; scores = {}
    for m, (g, k, _) in MODELS.items():
        if n - k - 1 < 1: continue
        r = fit(m, t, y)
        if r is None: continue
        fits[m] = r; scores[m] = aicc(2 * r.cost, n, k)
    scores = {m: v for m, v in scores.items() if np.isfinite(v)}
    sel = min(scores, key=scores.get); a = np.array(list(scores.values())); w = np.exp(-(a - a.min()) / 2); w /= w.sum()
    row = dict(selected=sel, exact=sel == law, cls=(sel in SINGLE_PHASE) == (law in SINGLE_PHASE),
               w_truth=float(w[list(scores).index(law)]) if law in scores else np.nan)
    if law in fits:
        r = fits[law]; se = wald_logse(r, n); err = r.x - np.log(p)
        for j in range(len(p)):
            row[f'p{j}_logerr'] = float(err[j]); row[f'p{j}_cover'] = bool(np.isfinite(se[j]) and abs(err[j]) <= 1.96 * se[j])
            row[f'p{j}_ident'] = bool(np.isfinite(se[j]) and se[j] < 0.354)
    for tag, mdl in [('M0', law), ('M1', sel)]:
        if mdl not in fits: continue
        g = MODELS[mdl][0]; ph = np.exp(fits[mdl].x)
        for gname, tg in GRIDS.items():
            row[f'{tag}_{gname}'] = float(np.sqrt(np.mean((g(tg, *ph) - f(tg, *p)) ** 2)))
    return row

def run_condition(args):
    truth, noise, place, n = args
    label = f'S1|{truth}|{noise}|{place}|{n}'; rng = np.random.default_rng(seed(label)); t = design(place, n)
    rows = []
    for i in range(REPS):
        r = one_replicate(rng, TRUTHS[truth], noise, t); r.update(truth=truth, noise=noise, placement=place, n=len(t), rep=i); rows.append(r)
    return rows

def loo_stability(args):
    truth, noise = args; label = f'LOO|{truth}|{noise}'; rng = np.random.default_rng(seed(label)); t = design('A_current', 8); law = TRUTHS[truth]; out = []
    for i in range(200):
        p = draw_truth(rng, law); y = add_noise(rng, MODELS[law][0](t, *p), noise)
        def select(tt, yy):
            sc = {}
            for m, (g, k, _) in MODELS.items():
                if len(tt) - k - 1 < 1: continue
                r = fit(m, tt, yy)
                if r is not None and np.isfinite(aicc(2 * r.cost, len(tt), k)): sc[m] = aicc(2 * r.cost, len(tt), k)
            return min(sc, key=sc.get)
        full = select(t, y); folds = [select(np.delete(t, j), np.delete(y, j)) for j in range(8)]
        out.append(dict(truth=truth, noise=noise, rep=i, full=full, agree_frac=float(np.mean([f == full for f in folds])),
                        fold_mode_share=float(max(folds.count(m) for m in set(folds)) / 8)))
    return out

def profile_cov(args):
    n, = args; label = f'PROF|{n}'; rng = np.random.default_rng(seed(label)); t = design('B_uniform', n); f = MODELS['PSO'][0]; out = []
    for i in range(100):
        p = draw_truth(rng, 'PSO'); y = add_noise(rng, f(t, *p), 'N3_add_0.03'); r0 = fit('PSO', t, y); sse0 = 2 * r0.cost; row = dict(n=n, rep=i)
        for j in range(2):
            grid = r0.x[j] + np.unique(np.r_[np.linspace(-3, 3, 121), np.linspace(-0.3, 0.3, 241)]); lr = []   # coarse + fine grid (implementation note 2)
            for v in grid:
                rr = least_squares(lambda q: f(t, *np.exp(np.insert(q, j, v))) - y, np.delete(r0.x, j), max_nfev=2000); lr.append(n * np.log(2 * rr.cost / sse0))
            ok = np.array(lr) <= 3.84; lo, hi = grid[ok].min(), grid[ok].max()
            row[f'p{j}_cover'] = bool(lo <= np.log(p[j]) <= hi); row[f'p{j}_closed'] = bool(not ok[0] and not ok[-1]); row[f'p{j}_logwidth'] = float(hi - lo)
        out.append(row)
    return out

if __name__ == '__main__':
    t0 = time.time()
    conds = [(tr, 'N3_add_0.03', pl, n) for tr in TRUTHS for pl in ['B_uniform', 'C_early', 'D_late'] for n in DENS]
    conds += [(tr, 'N3_add_0.03', 'A_current', 8) for tr in TRUTHS]
    conds += [(tr, nz, 'B_uniform', n) for tr in TRUTHS for nz in ['N1_add_0.01', 'N2_prop_3pct'] for n in DENS]
    conds += [(tr, nz, 'A_current', 8) for tr in TRUTHS for nz in ['N1_add_0.01', 'N2_prop_3pct']]
    print(f'{len(conds)} conditions x {REPS} replicates', flush=True)
    with Pool(2) as pool:
        rows = [r for res in pool.imap_unordered(run_condition, conds) for r in res]
        pd.DataFrame(rows).to_csv(OUT / 'stage1_replicates.csv.gz', index=False); print(f'main done {time.time() - t0:.0f}s', flush=True)
        loo = [r for res in pool.imap_unordered(loo_stability, [(tr, nz) for tr in TRUTHS for nz in ['N1_add_0.01', 'N2_prop_3pct', 'N3_add_0.03']]) for r in res]
        pd.DataFrame(loo).to_csv(OUT / 'stage1_loo_stability.csv.gz', index=False); print(f'loo done {time.time() - t0:.0f}s', flush=True)
        prof = [r for res in pool.imap_unordered(profile_cov, [(8,), (40,)]) for r in res]
        pd.DataFrame(prof).to_csv(OUT / 'stage1_profile_coverage.csv.gz', index=False)
    print(f'stage1 total {time.time() - t0:.0f}s', flush=True)
