"""Fig. 4: (a) paired dAUC of transferred LR minus T-LR for all 30 directed
pairs; (b) the same gap as a function of how many negative target labels both
learners may read (mean over the five sources, three random draws)."""
import numpy as np

from plot_style import GRAPH_COLOR, SHORT, WIDTH, load, panel, plt, save
import tt


def main():
    pairs, bud = load("transfer.json"), load("budget.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 1.85), gridspec_kw={"width_ratios": [1, 1.2]})
    m = np.full((6, 6), np.nan)
    for i, s in enumerate(tt.GRAPHS):
        for j, t in enumerate(tt.GRAPHS):
            if s != t:
                m[i, j] = pairs[f"{s}->{t}"]["LR"]["vs_tlr"]["delta"]
    ax.imshow(m, cmap="RdBu", vmin=-0.4, vmax=0.4)
    for i in range(6):
        for j in range(6):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.6,
                        color="w" if abs(m[i, j]) > 0.25 else "k")
    lab = [SHORT[g] for g in tt.GRAPHS]
    ax.set_xticks(range(6), lab, rotation=40, ha="right")
    ax.set_yticks(range(6), lab)
    ax.set_xlabel("target", labelpad=1)
    ax.set_ylabel("source", labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    panel(ax, "a")

    qs = sorted(bud[tt.GRAPHS[0]], key=float)
    for i, t in enumerate(tt.GRAPHS):
        x = [1 + bud[t][q]["n_neg_revealed"] for q in qs]
        y = [bud[t][q]["transfer_mean"] - bud[t][q]["T-LR"] for q in qs]
        bx.plot(x, y, "-o", ms=2.2, lw=0.9, color=GRAPH_COLOR[t], label=SHORT[t])
    bx.axhline(0, color="k", lw=0.5)
    bx.set_xscale("log")
    bx.set_xlabel("1 + negative target labels revealed", labelpad=1)
    bx.set_ylabel("AUC(transfer) $-$ AUC(T-LR)", labelpad=1)
    bx.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.5)
    panel(bx, "b")
    fig.tight_layout(pad=0.2, w_pad=0.8)
    save(fig, "transfer")


if __name__ == "__main__":
    main()
