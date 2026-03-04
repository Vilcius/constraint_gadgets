"""
plot_utils.py -- Shared matplotlib styling for constraint_gadget analysis.

Rose-pine moon palette consistent with existing analyze_results.py.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Rose-pine moon palette
# ---------------------------------------------------------------------------

_ROSE_PINE = {
    'base':        '#232136',
    'surface':     '#2a273f',
    'overlay':     '#393552',
    'muted':       '#6e6a86',
    'subtle':      '#908caa',
    'text':        '#e0def4',
    'love':        '#eb6f92',
    'gold':        '#f6c177',
    'rose':        '#ea9a97',
    'pine':        '#3e8fb0',
    'foam':        '#9ccfd8',
    'iris':        '#c4a7e7',
    'highlight_low':  '#2a283e',
    'highlight_med':  '#44415a',
    'highlight_high': '#56526e',
}

# Constraint-family colours
CONSTRAINT_COLORS = {
    'cardinality':     _ROSE_PINE['pine'],
    'knapsack':        _ROSE_PINE['gold'],
    'quadratic':       _ROSE_PINE['love'],
    'flow':            _ROSE_PINE['foam'],
    'assignment':      _ROSE_PINE['iris'],
    'subtour':         _ROSE_PINE['rose'],
    'independent_set': _ROSE_PINE['muted'],
    'unknown':         _ROSE_PINE['subtle'],
}

# Method colours
METHOD_COLORS = {
    'VCG':          _ROSE_PINE['pine'],
    'HybridQAOA':   _ROSE_PINE['iris'],
    'PenaltyQAOA':  _ROSE_PINE['love'],
}

# Angle-strategy colours
ANGLE_COLORS = {
    'QAOA':    _ROSE_PINE['gold'],
    'ma-QAOA': _ROSE_PINE['pine'],
}


def setup_style(moon: bool = True) -> None:
    """Apply rose-pine (moon variant) matplotlib style."""
    bg = _ROSE_PINE['base'] if moon else '#faf4ed'
    fg = _ROSE_PINE['text'] if moon else '#575279'
    grid = _ROSE_PINE['highlight_med'] if moon else '#dfdad9'

    mpl.rcParams.update({
        'figure.facecolor':  bg,
        'axes.facecolor':    bg,
        'axes.edgecolor':    fg,
        'axes.labelcolor':   fg,
        'axes.titlecolor':   fg,
        'axes.grid':         True,
        'grid.color':        grid,
        'grid.linewidth':    0.5,
        'xtick.color':       fg,
        'ytick.color':       fg,
        'text.color':        fg,
        'legend.facecolor':  _ROSE_PINE['surface'] if moon else '#fffaf3',
        'legend.edgecolor':  _ROSE_PINE['overlay'],
        'figure.dpi':        120,
        'savefig.dpi':       150,
        'savefig.facecolor': bg,
        'font.size':         11,
    })


def save_fig(fig: plt.Figure, path: str) -> None:
    """Save figure with tight_layout, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
