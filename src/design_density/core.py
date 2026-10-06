"""Core definitions for the design-density simulation study (see results/design_density/PROTOCOL.md).
Nothing here reads results; truths are parameterised from fits to the real kinetics sheet only."""
import numpy as np, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.optimize import least_squares

T_MAX = 170.0          # observed experimental window (min)
T_EXTRAP = 300.0       # extrapolation horizon used only for labelled extrapolation metrics
REAL_T = np.array([10, 20, 30, 50, 80, 110, 140, 170], float)

# ------------------------------------------------------------------ kinetic laws (all q(0)=0)
def pso(t, qe, k2): return k2 * qe ** 2 * t / (1 + k2 * qe * t)
def pfo(t, qe, k1): return qe * (1 - np.exp(-k1 * t))
MODELS = {
    'PSO':        (lambda t, a, b: pso(t, a, b), 2, [[1.2, .1], [1.0, .02]]),
    'PFO':        (lambda t, a, b: pfo(t, a, b), 2, [[1.1, .08], [1.0, .02]]),
    'Elovich':    (lambda t, a, b: np.log1p(a * b * t) / b, 2, [[1., 5.], [10., 3.]]),
    'PSO+sqrt':   (lambda t, a, b, c: pso(t, a, b) + c * np.sqrt(t), 3, [[.8, .3, .02], [.6, 1., .03]]),
    'TwoSitePSO': (lambda t, a, b, c, d: pso(t, a, b) + pso(t, c, d), 4, [[.8, 1., .6, .01], [.85, .5, .8, .003]]),
    'TwoSitePFO': (lambda t, a, b, c, d: pfo(t, a, b) + pfo(t, c, d), 4, [[.8, .2, .5, .01], [.85, .1, .8, .003]]),
}
SINGLE_PHASE = {'PSO', 'PFO'}          # phase class used for class-level recovery
SLOW_PHASE = {'Elovich', 'PSO+sqrt', 'TwoSitePSO', 'TwoSitePFO'}

def fit(name, t, y):
    f, k, starts = MODELS[name]; best = None
    for p0 in starts:
        try:
            r = least_squares(lambda p: f(t, *np.exp(p)) - y, np.log(p0), method='trf', max_nfev=3000, x_scale='jac')
            if np.isfinite(r.cost) and np.all(np.isfinite(r.x)) and (best is None or r.cost < best.cost): best = r   # non-finite fits count as failures
        except Exception:
            pass
    return best

def aicc(sse, n, k):
    if n - k - 1 < 1: return np.nan
    return n * np.log(max(sse, 1e-300) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def wald_logse(r, n):
    """SE of log-parameters from the Gauss-Newton covariance; nan if singular."""
    k = len(r.x)
    if n - k < 1: return np.full(k, np.nan)
    JTJ = r.jac.T @ r.jac
    try:
        cov = np.linalg.inv(JTJ) * (2 * r.cost / (n - k))
        d = np.diag(cov); return np.where(d > 0, np.sqrt(np.abs(d)), np.nan)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

# ------------------------------------------------------------------ sampling designs on (0, 170]
def design(kind, n):
    if kind == 'A_current':
        return REAL_T.copy()
    u = np.arange(1, n + 1) / n
    if kind == 'B_uniform':  return T_MAX * u
    if kind == 'C_early':    return np.geomspace(2.0, T_MAX, n)
    if kind == 'D_late':     return T_MAX * np.sqrt(u)
    raise ValueError(kind)

# ------------------------------------------------------------------ noise scenarios
NOISE = {'N1_add_0.01': ('add', 0.01), 'N2_prop_3pct': ('prop', 0.03), 'N3_add_0.03': ('add', 0.03)}
def add_noise(rng, q, scen):
    kind, s = NOISE[scen]
    return q + rng.normal(0, s, q.shape) if kind == 'add' else q * (1 + rng.normal(0, s, q.shape))
