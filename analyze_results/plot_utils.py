"""
plot_utils.py -- Shared matplotlib styling for constraint_gadget analysis.

Light theme suitable for papers and GitHub.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Color palette (paper-friendly, light background)
# ---------------------------------------------------------------------------

_ROSE_PINE = {
    'base':        '#ffffff',   # white background
    'surface':     '#f6f8fa',   # very light gray panels
    'overlay':     '#e8eaed',   # light gray overlay
    'muted':       '#6e7681',   # muted text
    'subtle':      '#57606a',   # secondary text
    'text':        '#24292f',   # near-black primary text
    'love':        '#cf222e',   # red   — infeasible / PenaltyQAOA
    'gold':        '#b08800',   # amber — struct✓ / penalty✗
    'rose':        '#e85c8a',   # pink  — struct✗ / penalty✓
    'pine':        '#1f6feb',   # blue  — feasible / HybridQAOA / QAOA
    'foam':        '#1a7f37',   # green — optimal
    'iris':        '#8250df',   # purple — ma-QAOA / HybridQAOA
    'highlight_low':  '#f6f8fa',
    'highlight_med':  '#e8eaed',
    'highlight_high': '#d0d7de',
}

# Constraint-family colours
CONSTRAINT_COLORS = {
    'cardinality':        _ROSE_PINE['pine'],
    'knapsack':           _ROSE_PINE['gold'],
    'quadratic':          _ROSE_PINE['love'],
    'quadratic_knapsack': _ROSE_PINE['rose'],
    'flow':               _ROSE_PINE['foam'],
    'assignment':         _ROSE_PINE['iris'],
    'subtour':            _ROSE_PINE['muted'],
    'independent_set':    _ROSE_PINE['subtle'],
    'unknown':            _ROSE_PINE['muted'],
}

# Method colours
METHOD_COLORS = {
    'VCG':          _ROSE_PINE['pine'],
    'HybridQAOA':   '#9f7aea',   # amethyst / lavender
    'PenaltyQAOA':  '#4fc3f7',   # crystal blue / cyan
}

# Angle-strategy colours
ANGLE_COLORS = {
    'QAOA':    _ROSE_PINE['gold'],
    'ma-QAOA': _ROSE_PINE['pine'],
}


def setup_style() -> None:
    """Apply clean light matplotlib style suitable for papers and GitHub."""
    bg   = _ROSE_PINE['base']
    fg   = _ROSE_PINE['text']
    grid = _ROSE_PINE['highlight_high']

    mpl.rcParams.update({
        'figure.facecolor':  bg,
        'axes.facecolor':    bg,
        'axes.edgecolor':    _ROSE_PINE['highlight_high'],
        'axes.labelcolor':   fg,
        'axes.titlecolor':   fg,
        'axes.grid':         True,
        'grid.color':        grid,
        'grid.linewidth':    0.5,
        'grid.linestyle':    '--',
        'xtick.color':       fg,
        'ytick.color':       fg,
        'text.color':        fg,
        'legend.facecolor':  _ROSE_PINE['surface'],
        'legend.edgecolor':  _ROSE_PINE['highlight_high'],
        'legend.framealpha': 1.0,
        'figure.dpi':        120,
        'savefig.dpi':       150,
        'savefig.facecolor': bg,
        'font.size':         11,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.formatter.use_mathtext':   False,
        'axes.formatter.limits':         (-9999, 9999),
    })


def save_fig(fig: plt.Figure, path: str) -> None:
    """Save figure with tight_layout, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
