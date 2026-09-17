from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / 'figures'

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'text.color': '#0b0b0b',
})

fig, ax = plt.subplots(figsize=(7.0, 4.75))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1)

INK = '#0b0b0b'
SEC = '#52514e'
RULE = '#c3c2b7'

# Notation preamble
notation = (r"$\mathrm{Notation:}\ \ \hat{q}_\theta(t,C_0,D)\ \mathrm{is\ the\ network\ output\ (real\ units);}\ "
            r"\hat{q}_{e}(C_0,D)\ \mathrm{is\ the\ Freundlich/mass\!-\!balance\ equilibrium\ capacity\ (Eq.\ 6).}$")
ax.text(0.03, 0.965, notation, fontsize=9.2, color=SEC, ha='left', va='top')
ax.plot([0.03, 0.97], [0.905, 0.905], color=RULE, lw=0.8)

eqs = [
    (r"$\mathcal{L}_{data} = \frac{1}{N}\sum_{i=1}^{N}(\hat{q}_\theta(t_i,C_{0,i},D_i) - q_i)^{2}$",
     "(1)"),
    (r"$\mathcal{L}_{PSO} = \frac{1}{M}\sum_{j=1}^{M}(\frac{\partial \hat{q}_\theta}{\partial t}|_{j} - \hat{k}_2(\hat{q}_{e,j}-\hat{q}_{\theta,j})^{2})^{2}$",
     "(2)"),
    (r"$\mathcal{L}_{IC} = \frac{1}{M}\sum_{j=1}^{M}(\hat{q}_\theta(0,C_{0,j},D_j))^{2}$",
     "(3)"),
    (r"$\mathcal{L}_{eq} = \frac{1}{M}\sum_{j=1}^{M}(\hat{q}_\theta(400,C_{0,j},D_j) - \hat{q}_{e,j})^{2}$",
     "(4)"),
    (r"$\mathcal{L}_{total} = 180\,\mathcal{L}_{data} + 1\,\mathcal{L}_{PSO} + 35\,\mathcal{L}_{IC} + 8\,\mathcal{L}_{eq}$",
     "(5)"),
    (r"$\hat{q}_{e} = \hat{K}_f(C_0 - \hat{q}_{e}/D)^{\hat{n}}$",
     "(6)"),
]

y0 = 0.80
dy = 0.135
for (eq, num), y in zip(eqs, [y0 - i*dy for i in range(len(eqs))]):
    ax.text(0.05, y, eq, fontsize=12.5, color=INK, ha='left', va='center')
    ax.text(0.965, y, num, fontsize=11, color=SEC, ha='right', va='center')

ax.text(0.05, y0 - (len(eqs) - 1) * dy - 0.075,
        r"$\mathrm{solved\ self\!-\!consistently\ for\ }\hat{q}_e\mathrm{\ by\ damped\ fixed\!-\!point\ iteration.}$",
        fontsize=8.8, color=SEC, ha='left', va='center')

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / 'fig_pinn3d_equations.png', dpi=400, bbox_inches='tight', facecolor='white')
print('saved equations image')
