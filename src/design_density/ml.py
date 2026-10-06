"""Stage-2 learners for the design-density study (pre-specified architectures; see PROTOCOL.md)."""
import numpy as np, torch, torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, RBF, WhiteKernel
torch.set_num_threads(1)

def pinn_fit(t, y, hidden, epochs=4000, seed=0, lr=0.005, w=(180.0, 1.0, 35.0), n_coll=100, t_coll_max=200.0):
    """Kinetic arm of the existing PINN3D formulation: tanh MLP with LayerNorm on standardized t -> standardized q,
    trainable k2 (softplus + 1e-3 floor, as in train.py) and qe (softplus; qe replaces the Freundlich/mass-balance
    closure, which needs the isotherm arm), losses = w_data*MSE(standardized) + w_pso*PSO residual (real units, autograd,
    chain rule) + w_ic*q(0)^2 (real units). No equilibrium anchor (it requires the isotherm arm)."""
    torch.manual_seed(seed)
    tm, ts = t.mean(), t.std() + 1e-12; ym, ys = y.mean(), y.std() + 1e-12
    layers, d = [], 1
    for h in hidden: layers += [torch.nn.Linear(d, h), torch.nn.LayerNorm(h), torch.nn.Tanh()]; d = h
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
    rk2 = torch.nn.Parameter(torch.tensor([-3.0])); rqe = torch.nn.Parameter(torch.tensor([float(np.log(np.expm1(max(y.max(), 0.1))))]))
    opt = torch.optim.Adam(list(net.parameters()) + [rk2, rqe], lr=lr)
    X = torch.tensor(((t - tm) / ts)[:, None], dtype=torch.float32); Y = torch.tensor(((y - ym) / ys)[:, None], dtype=torch.float32)
    Xc = torch.tensor(((np.linspace(0, t_coll_max, n_coll) - tm) / ts)[:, None], dtype=torch.float32, requires_grad=True)
    X0 = torch.tensor([[(0 - tm) / ts]], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        dl = torch.mean((net(X) - Y) ** 2)
        qc = net(Xc); dq = torch.autograd.grad(qc.sum(), Xc, create_graph=True)[0] * ys / ts
        k2 = F.softplus(rk2) + 1e-3; qe = F.softplus(rqe)
        pl = torch.mean((dq - k2 * (qe - (qc * ys + ym)) ** 2) ** 2)
        il = torch.mean((net(X0) * ys + ym) ** 2)
        (w[0] * dl + w[1] * pl + w[2] * il).backward(); opt.step()
    def pred(tq):
        with torch.no_grad(): return net(torch.tensor(((np.asarray(tq) - tm) / ts)[:, None], dtype=torch.float32)).numpy().ravel() * ys + ym
    return pred, dict(k2=float(F.softplus(rk2) + 1e-3), qe=float(F.softplus(rqe)), n_params=sum(p.numel() for p in net.parameters()) + 2)

def nn_residual(t, resid, seeds=range(5), epochs=2000):
    """Tiny residual MLP 1-4-1 tanh (13 weights): Delta(t) = (t/170) * MLP(t/170); weight decay 1e-3; seed-ensemble mean."""
    xs = torch.tensor(t[:, None] / 170., dtype=torch.float32); ys = torch.tensor(resid[:, None], dtype=torch.float32); nets = []
    for s in seeds:
        torch.manual_seed(s); net = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
        for _ in range(epochs):
            opt.zero_grad(); l = torch.mean((xs * net(xs) - ys) ** 2); l.backward(); opt.step()
        nets.append(net)
    def pred(tq):
        x = torch.tensor(np.asarray(tq)[:, None] / 170., dtype=torch.float32)
        with torch.no_grad(): return np.mean([(x * n(x)).numpy().ravel() for n in nets], 0)
    return pred

def gp_residual(t, resid):
    """GP residual, RBF length-scale by marginal likelihood within [5, 500] min, white noise; (0, 0) anchored."""
    g = GaussianProcessRegressor(Ck(max(np.var(resid), 1e-6), (1e-8, 10)) * RBF(50., (5., 500.)) + WhiteKernel(1e-3, (1e-8, 1e-1)),
                                 normalize_y=False, n_restarts_optimizer=3, random_state=0)
    g.fit(np.r_[0., t][:, None], np.r_[0., resid])
    return lambda tq: g.predict(np.asarray(tq)[:, None])
