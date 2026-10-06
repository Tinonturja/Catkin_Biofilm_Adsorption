"""One figure style for every plot in this repository.

Import this module before plotting:

    import figstyle as fs
    fs.apply()
    fig, ax = fs.figure(width='double', height_mm=120, nrows=2, ncols=2)
    ...
    fs.save(fig, OUT / 'name')          # writes name.pdf (vector) and name.png (600 dpi)

Roles are fixed across figures so that a colour or marker always means the same thing.
Colours are a three-hue subset of the Okabe-Ito palette plus neutrals. The subset was checked for
separation under deuteranopia, protanopia and tritanopia; every series also differs by line style or
marker, so no figure relies on colour alone.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTH_MM = {'single': 90, 'onehalf': 140, 'double': 180}

# ---------------------------------------------------------------- colours by role
INK = '#000000'          # measured data, axes, text
BLUE = '#0072B2'         # the preferred description; first category
VERMILLION = '#D55E00'   # second category
GREEN = '#009E73'        # third category
GREY_DARK = '#4D4D4D'    # conventional / baseline models
GREY_MID = '#8C8C8C'     # second baseline
GREY_LIGHT = '#D9D9D9'   # distributions, reference bands
BAND = '#EFEFEF'         # shaded reference regions

# kinetic and isotherm laws: (colour, line style, line width)
LAW = {
    'Burst + sqrt(t)': (BLUE, '-', 1.5),
    'Power law':       (GREEN, (0, (5, 1.5)), 1.2),
    'Elovich':         (VERMILLION, (0, (3, 1, 1, 1)), 1.2),
    'PSO':             (GREY_DARK, (0, (4, 2)), 1.1),
    'PFO':             (GREY_MID, (0, (1, 1.5)), 1.1),
    'Freundlich':      (BLUE, '-', 1.5),
    'Linear':          (GREY_DARK, (0, (4, 2)), 1.1),
}
# sampling designs in the simulation: (label, colour, line style, marker)
DESIGN = {
    'B_uniform': ('uniform', BLUE, '-', 'o'),
    'C_early':   ('early-enriched', GREEN, '-', 's'),
    'D_late':    ('late-enriched', VERMILLION, (0, (4, 2)), '^'),
}
# up to four parameters of one law: (colour, marker, line style)
PARAM = [(BLUE, 'o', '-'), (VERMILLION, 's', (0, (4, 2))), (GREEN, '^', (0, (1, 1.5))), (GREY_DARK, 'D', (0, (3, 1, 1, 1)))]
# ordered quantity (number of observations 8 < 16 < 40): one hue, light to dark
SEQ3 = ['#9ECAE1', '#4292C6', '#084594']

MEASURED = dict(marker='o', ms=4.2, mfc=INK, mec='white', mew=0.6, ls='none', zorder=5)


def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'legend.frameon': False, 'legend.handlelength': 2.4,
        'mathtext.default': 'regular',
        'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.grid': False, 'grid.color': '#E6E6E6', 'grid.linewidth': 0.5,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'standard',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def figure(width='double', height_mm=90, **kw):
    """Figure at its printed size, so text sizes on screen equal text sizes in the journal.

    Files are saved at exactly this size (no automatic cropping). Set margins with fig.subplots_adjust.
    """
    w = WIDTH_MM[width] if isinstance(width, str) else width
    return plt.subplots(figsize=(w * MM, height_mm * MM), **kw)


def panel(ax, letter, title=None, x=0.0):
    """Bold panel letter at the top-left corner, optional plain title after it."""
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
    if title:
        ax.text(x + 0.075, 1.04, title, transform=ax.transAxes, fontsize=8, va='bottom', ha='left')


def save(fig, stem):
    """Vector PDF for submission and a 600 dpi PNG for quick viewing."""
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=600)
    plt.close(fig)
