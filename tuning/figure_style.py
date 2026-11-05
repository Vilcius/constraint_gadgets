"""Shared publication style for the tuning and final-experiment notebooks."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import warnings

import matplotlib as mpl


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

METHOD_COLORS = {
    "PC-QAOA": PALETTE["blue_main"],
    "PC-QAOA (StatePrep)": PALETTE["violet"],
    "PenaltyQAOA": PALETTE["red_strong"],
    "QAOA+": PALETTE["teal"],
}


def apply_publication_style():
    """Apply the house style while retaining Computer Modern typography."""
    mpl.rcParams.update({
        # Matplotlib ships the original Computer Modern Roman glyphs as
        # cmr10.ttf.  Using them directly avoids a PGF/LaTeX failure changing
        # the font partway through a notebook.
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelsize": 12,
        "axes.labelpad": 6,
        "axes.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "errorbar.capsize": 3.0,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,
        "legend.frameon": False,
        "legend.borderaxespad": 0.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "text.usetex": False,
    })


def polish_figure(figure):
    """Apply consistent print-safe details after a figure has been drawn."""
    from matplotlib.collections import PathCollection

    for axis in figure.axes:
        # Colorbar axes have no data title/labels to polish like a plot axis.
        is_colorbar = axis.get_label() == "<colorbar>"
        if not is_colorbar:
            axis.spines["left"].set_linewidth(1.4)
            axis.spines["bottom"].set_linewidth(1.4)
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.55)
            axis.tick_params(direction="out", width=1.1, length=4)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
        for line in axis.lines:
            line.set_linewidth(max(1.8, line.get_linewidth()))
            if line.get_marker() not in (None, "None", ""):
                line.set_markersize(max(6.0, line.get_markersize()))
                line.set_markeredgewidth(max(0.7, line.get_markeredgewidth()))
        for patch in axis.patches:
            # Adds separation in print without changing the encoded fill color.
            if hasattr(patch, "get_width") and hasattr(patch, "get_height"):
                patch.set_edgecolor("white")
                patch.set_linewidth(max(0.7, patch.get_linewidth()))
        for collection in axis.collections:
            if isinstance(collection, PathCollection):
                collection.set_linewidths(0.55)
    return figure


class FigureWriter:
    """Computer Modern PDF export with matching 300-DPI PNG previews."""

    def __init__(self, directory, latex=True, display_inline=False):
        apply_publication_style()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Keep one deterministic renderer for every figure.  The bundled cmr10
        # font is Computer Modern and does not depend on external TeX packages.
        self.latex = False
        self.display_inline = display_inline
        self.font_status = "Matplotlib Computer Modern (cmr10)"

    def save(self, figure, name):
        import matplotlib.pyplot as plt

        polish_figure(figure)
        pdf = self.directory / f"{name}.pdf"
        png = self.directory / f"{name}.png"
        figure.tight_layout(pad=1.2)
        figure.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
        figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
        (self.directory / "font_status.json").write_text(
            json.dumps({"font": self.font_status}, indent=2) + "\n"
        )
        plt.close(figure)
        if self.display_inline:
            from IPython.display import display, Image
            display(Image(filename=str(png), width=900))
        return png
