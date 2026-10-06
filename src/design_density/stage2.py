"""Stage 2 (learners) of the design-density study. Implements PROTOCOL.md section 7.
Run from the project root after stage 1: python src/design_density/stage2.py"""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import MODELS, fit, aicc, design, add_noise
from stage1 import TRUTHS, GRIDS, draw_truth, seed, OUT
from ml import pinn_fit, nn_residual, gp_residual

TRUTHS2 = ['T1_PSO', 'T3_Elovich', 'T4_TwoSitePFO']
DESIGNS2 = [('A_current', 8), ('B_uniform', 16), ('B_uniform', 40)]
REPS2 = 30

def task(args):
    truth, place, n, i = args
    rng = np.random.default_rng(seed(f'S2|{truth}|{place}|{n}') + i)
    law = TRUTHS[truth]; f = MODELS[law][0]; t = design(place, n); p = draw_truth(rng, law); y = add_noise(rng, f(t, *p), 'N3_add_0.03')
    fits, sc = {}, {}
    for m, (g, k, _) in MODELS.items():
        if len(t) - k - 1 < 1: continue
        r = fit(m, t, y)
        if r is not None and np.isfinite(aicc(2 * r.cost, len(t), k)): fits[m] = r; sc[m] = aicc(2 * r.cost, len(t), k)
    sel = min(sc, key=sc.get)
    m1 = lambda tq: MODELS[sel][0](np.asarray(tq), *np.exp(fits[sel].x))
    m0 = lambda tq: MODELS[law][0](np.asarray(tq), *np.exp(fits[law].x))
    resid = y - m1(t)
    nres = nn_residual(t, resid); gres = gp_residual(t, resid)
    pinn_big, info_b = pinn_fit(t, y, (32, 32), seed=i); pinn_small, info_s = pinn_fit(t, y, (8, 8), seed=i)
    learners = {'M0': m0, 'M1': m1, 'M2_PINN_32x32': pinn_big, 'M2s_PINN_8x8': pinn_small,
                'M3_NNresid': lambda tq: m1(tq) + nres(tq), 'M4_GPresid': lambda tq: m1(tq) + gres(tq)}
    row = dict(truth=truth, placement=place, n=n, rep=i, selected=sel)
    for name, fn in learners.items():
        for gname, tg in GRIDS.items():
            row[f'{name}_{gname}'] = float(np.sqrt(np.mean((fn(tg) - f(tg, *p)) ** 2)))
    if law == 'PSO':
        row.update(PINN_k2_logerr=float(np.log(info_b['k2'] / p[1])), PINN_qe_logerr=float(np.log(info_b['qe'] / p[0])),
                   PINNs_k2_logerr=float(np.log(info_s['k2'] / p[1])), PINNs_qe_logerr=float(np.log(info_s['qe'] / p[0])))
    row['PINN_params'] = info_b['n_params']; row['PINNs_params'] = info_s['n_params']
    return row

if __name__ == '__main__':
    t0 = time.time(); tasks = [(tr, pl, n, i) for tr in TRUTHS2 for pl, n in DESIGNS2 for i in range(REPS2)]
    print(f'{len(tasks)} stage-2 replicates', flush=True); rows = []
    with Pool(2) as pool:
        for k, r in enumerate(pool.imap_unordered(task, tasks)):
            rows.append(r)
            if (k + 1) % 30 == 0: print(f'{k + 1} done {time.time() - t0:.0f}s', flush=True); pd.DataFrame(rows).to_csv(OUT / 'stage2_replicates.csv.gz', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'stage2_replicates.csv.gz', index=False); print(f'stage2 total {time.time() - t0:.0f}s', flush=True)
