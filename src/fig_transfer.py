"""Fig. 4 (forecast contract): (a) paired dAUC of transferred LR minus T-LR for
all 30 directed pairs; (b) positive control, transfer between two random user
halves of one platform minus T-LR, with the source half on the target's window
(h=0.4) or on its whole history (h=1); bars = mean of 3 splits, whiskers = range."""
import numpy as np

from plot_style import C_SAME, SHORT, WIDTH, load, panel, plt, save
import tt


def main():
    pairs = load("strict/transfer.json")
    c04, c10 = load("strict/control_h0.4.json"), load("strict/control.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})
    m = np.full((6, 6), np.nan)
    for i, s in enumerate(tt.GRAPHS):
        for j, t in enumerate(tt.GRAPHS):
            if s != t:
                m[i, j] = pairs[f"{s}->{t}"]["LR"]["vs_tlr"]["delta"]
    ax.imshow(m, cmap="RdBu", vmin=-0.4, vmax=0.4)
    for i in range(6):
        for j in range(6):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.2f}".replace("0.", ".").replace("-", "\u2212"), ha="center", va="center", fontsize=5.6,
                        color="w" if abs(m[i, j]) > 0.25 else "k")
    lab = [SHORT[g] for g in tt.GRAPHS]
    ax.set_xticks(range(6), lab, rotation=40, ha="right")
    ax.set_yticks(range(6), lab)
    ax.set_xlabel("target", labelpad=1)
    ax.set_ylabel("source", labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    panel(ax, "a")

    g = tt.GRAPHS
    x = np.arange(len(g))
    for off, ctrl, col, name in ((-0.2, c04, C_SAME, "source half on the target's window ($h=0.4$)"),
                                 (0.2, c10, "#bbbbbb", "whole history ($h=1$)")):
        d = np.array([[r["lr"]["delta"] for r in ctrl[t]] for t in g])
        mean = d.mean(1)
        bx.bar(x + off, mean, 0.4, color=col, label=name)
        bx.errorbar(x + off, mean, yerr=[mean - d.min(1), d.max(1) - mean], fmt="none", ecolor="k",
                    elinewidth=0.6, capsize=1.2)
    bx.axhline(0, color="k", lw=0.5)
    bx.set_xticks(x, lab, rotation=40, ha="right")
    bx.set_ylabel("$\\Delta$AUC, half to half", labelpad=1)
    bx.legend(frameon=False, loc="upper right", fontsize=5.5, handlelength=1.0)
    panel(bx, "b")
    fig.tight_layout(pad=0.2, w_pad=1.0)
    save(fig, "transfer")


if __name__ == "__main__":
    main()
