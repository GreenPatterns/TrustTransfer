"""Fig. 5: (a) positive control, transfer between node-disjoint halves of one
platform minus T-LR, with the source half on the target's window (h=0.4) or
on its full history (h=1); bars = mean of 3 random splits, whiskers = range.
(b) source history: mean paired dAUC (transferred LR minus T-LR) over all,
same-type and cross-type pairs."""
import numpy as np

from plot_style import C_CROSS, C_REF, C_SAME, SAME_TYPE, SHORT, WIDTH, load, panel, plt, save
import tt


def main():
    c04, c10, pairs = load("control_h0.4.json"), load("control.json"), load("transfer.json")
    g = tt.GRAPHS
    x = np.arange(len(g))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 1.7), gridspec_kw={"width_ratios": [1.5, 1]})
    for off, ctrl, col, lab in ((-0.2, c04, C_SAME, "target window ($h{=}0.4$)"),
                                (0.2, c10, "#bbbbbb", "full history ($h{=}1$)")):
        d = np.array([[r["lr"]["delta"] for r in ctrl[t]] for t in g])
        mean = d.mean(1)
        ax.bar(x + off, mean, 0.4, color=col, label=lab)
        ax.errorbar(x + off, mean, yerr=[mean - d.min(1), d.max(1) - mean], fmt="none", ecolor="k", elinewidth=0.6,
                    capsize=1.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, [SHORT[t] for t in g], rotation=30, ha="right")
    ax.set_ylabel("$\\Delta$AUC vs. T-LR")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.6,
              handlelength=1.0, columnspacing=0.8)
    panel(ax, "a", "tr")

    hs = ["0.2", "0.5", "0.8", "1.0"]
    groups = (("all 30 pairs", C_REF, lambda p: True), ("same type (4)", C_SAME, lambda p: p in SAME_TYPE),
              ("cross type (26)", C_CROSS, lambda p: p not in SAME_TYPE))
    for lab, col, keep in groups:
        y = [np.mean([v["history"][h]["delta_tlr"] for k, v in pairs.items() if keep(tuple(k.split("->")))])
             for h in hs]
        bx.plot([float(h) for h in hs], y, "-o", ms=2.5, color=col, label=lab)
    bx.axhline(0, color="k", lw=0.5)
    bx.set_xlabel("source history used, $h$")
    bx.set_ylabel("mean $\\Delta$AUC vs. T-LR")
    bx.set_xticks([0.2, 0.5, 0.8, 1.0])
    bx.legend(frameon=False, fontsize=5.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
    panel(bx, "b", "tl")
    fig.tight_layout(pad=0.2, w_pad=1.0)
    save(fig, "controls")


if __name__ == "__main__":
    main()
