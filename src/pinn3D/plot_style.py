"""
Shared plotting style for the Catkin PINN3D manuscript figures.
Palette + mark conventions adapted from the dataviz skill's validated
default palette (references/palette.md), translated to static/print
(matplotlib) use for a clean, consistent, journal-style look across all
PINN3D figures in the paper.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- Categorical palette (validated, colorblind-safe adjacent ordering) ----
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'
YELLOW = '#c98500'   # dark-mode step used here for slightly better print contrast
MAGENTA = '#c2447a'  # deepened for print contrast (light-mode magenta is sub-3:1)
GREEN = '#008300'
VIOLET = '#4a3aa7'
RED = '#c9302c'      # deepened red for print

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# ---- Sequential blue ramp (light -> dark), for heatmaps ----
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']

# ---- Chart chrome / ink ----
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
SURFACE = '#ffffff'

# Role assignments used consistently across all PINN3D figures
COLOR_DATA = INK_PRIMARY       # experimental data points
COLOR_PFO = BLUE               # classical PFO fit
COLOR_PSO = ORANGE             # classical PSO / Freundlich fit
COLOR_PINN = GREEN             # PINN3D ensemble mean
COLOR_BAND = GREEN             # PINN3D uncertainty band
COLOR_KINETICS = GREEN
COLOR_ISOTHERM = RED
COLOR_CEILING = RED


def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10.5,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10.5,
        'axes.labelcolor': INK_PRIMARY,
        'axes.edgecolor': BASELINE,
        'axes.linewidth': 0.9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': INK_SECONDARY,
        'ytick.color': INK_SECONDARY,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'grid.color': GRIDLINE,
        'grid.linewidth': 0.7,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.8,
        'lines.markersize': 5.5,
        'mathtext.fontset': 'stixsans',
    })


def panel_label(ax, letter, x=-0.14, y=1.06):
    """Bold panel letter (A, B, C...) in the Nature-figure convention."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
             va='bottom', ha='left', color=INK_PRIMARY)


def style_axes(ax, light_grid_axis=None):
    """Recessive gridlines on request; muted baseline; no chartjunk."""
    ax.tick_params(length=3, width=0.9, colors=INK_SECONDARY)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color(BASELINE)
    if light_grid_axis:
        ax.grid(axis=light_grid_axis, color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
