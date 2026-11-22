"""Shared matplotlib style for the paper's figures: Computer Modern serif to
match LNCS body text, sizes for a 122 mm text width, vector PDF output."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tt  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

FIG = os.path.join(tt.ROOT, "figures")
WIDTH = 4.8                       # LNCS text width in inches
SHORT = {"otc": "OTC", "alpha": "Alpha", "epinions": "Epinions", "slashdot": "Slashdot",
         "wikirfa": "RfA", "reddit": "Reddit"}
SAME_TYPE = {("otc", "alpha"), ("alpha", "otc"), ("epinions", "slashdot"), ("slashdot", "epinions")}
C_SAME, C_CROSS, C_REF, C_LEAK = "#1f77b4", "#ff7f0e", "#333333", "#d62728"
GRAPH_COLOR = dict(zip(tt.GRAPHS, ["#7570b3", "#e7298a", "#1b9e77", "#a6761d", "#66a61e", "#666666"]))


def load(name):
    with open(os.path.join(tt.RESULTS, name)) as f:
        return json.load(f)


def panel(ax, letter, where="outside"):
    """Panel letter: above the top-left corner, or inside the top-right / top-left corner."""
    x, y, ha, va = {"outside": (-0.02, 1.04, "right", "bottom"), "tr": (0.98, 0.98, "right", "top"),
                    "tl": (0.03, 0.98, "left", "top")}[where]
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=7, fontweight="bold", ha=ha, va=va)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), metadata={"CreationDate": None})
    print("saved", name)
