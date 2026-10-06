"""Refit the truth parameters of the design-density simulation and compare them with the stored file.

The simulation (src/design_density/stage1.py) reads results/design_density/truth_fits.json. This script
refits the five generating laws to the kinetic sheet, writes results/final_analysis/truth_fits_regenerated.json
and prints the difference from the stored file. The stored file is not overwritten, because the archived
simulation results were produced from it.

Run from the project root:  python src/final_analysis/make_truth_fits.py
"""
import json
import sys
import warnings

import numpy as np
from scipy.optimize import least_squares

from common import ROOT, OUT, load_kinetics

warnings.filterwarnings('ignore')
sys.path.insert(0, str(ROOT / 'src' / 'design_density'))
from core import MODELS, wald_logse  # noqa: E402

t, y = load_kinetics()
n = len(t)
TRUTHS = ['PSO', 'PFO', 'Elovich', 'PSO+sqrt', 'TwoSitePFO']
out = {}
for name in TRUTHS:
    f, k, _ = MODELS[name]
    best = None
    for s in range(200):
        p0 = np.random.default_rng(s).normal(0, 2, k) + np.log([1, .05, 1, .01][:k])
        try:
            r = least_squares(lambda p: f(t, *np.exp(p)) - y, p0, max_nfev=4000)
        except Exception:
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost - 1e-14):
            best = r
    out[name] = dict(logp=best.x.tolist(), p=np.exp(best.x).tolist(),
                     logse=wald_logse(best, n).tolist(), rmse=float(np.sqrt(2 * best.cost / n)))

json.dump(out, open(OUT / 'truth_fits_regenerated.json', 'w'), indent=1)
print('wrote', OUT / 'truth_fits_regenerated.json')

stored = ROOT / 'results' / 'design_density' / 'truth_fits.json'
if stored.exists():
    S = json.load(open(stored))
    print('comparison with the stored file (parameter order may differ for two-site laws, so values are sorted):')
    for name in TRUTHS:
        if name not in S:
            print(f'  {name}: not in stored file')
            continue
        dp = np.max(np.abs(np.sort(S[name]['logp']) - np.sort(out[name]['logp'])))
        print(f"  {name:<11s} max |d log p| = {dp:.1e}   rmse stored {S[name]['rmse']:.5f}  fresh {out[name]['rmse']:.5f}")
else:
    print('no stored truth_fits.json found; copy the regenerated file to results/design_density/truth_fits.json')
