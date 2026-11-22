"""Fig. 5: few-shot transfer at fixed negative-label budgets (forecast contract).
One panel per target: test AUC against the number k of negative target labels
revealed (with the matching share of positives), for the target-only model and
four transfer variants averaged over the five sources; 10 draws per budget,
band = one standard deviation of T-LR over draws. Dashed: T-LR with the whole prefix."""
import numpy as np

from plot_style import C_REF, SHORT, WIDTH, load, plt, save
import tt

METHODS = (("ZS-LR", "#ff7f0e", "zero-shot LR"), ("WS-LR", "#1f77b4", "warm-start LR"),
           ("Pool-LR", "#2ca02c", "pooled LR"), ("WS-GBDT", "#9467bd", "warm-start GBDT"))


def main(name="strict/fewshot.json", out="fewshot"):
    fs, targets = load(name), load(name.replace("fewshot", "targets"))
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 1.95), sharex=True)
    for ax, t in zip(axes.ravel(), tt.GRAPHS):
        ks = sorted(fs[t]["budgets"], key=int)
        x = np.arange(len(ks))
        tl = np.array([[r["T-LR"] for r in fs[t]["budgets"][k]] for k in ks])
        ax.fill_between(x, tl.mean(1) - tl.std(1), tl.mean(1) + tl.std(1), color=C_REF, alpha=0.12, lw=0)
        ax.plot(x, tl.mean(1), "-o", color=C_REF, ms=2.2, lw=1.1, label="T-LR (target only)")
        for m, col, lab in METHODS:
            y = [np.mean([np.mean([v for n, v in r.items() if n.startswith(m + "|")])
                          for r in fs[t]["budgets"][k]]) for k in ks]
            ax.plot(x, y, "-o", color=col, ms=1.8, lw=0.8, label=lab)
        ax.axhline(targets[t]["T-LR"]["auc"], color=C_REF, lw=0.6, ls="--")
        ax.axhline(0.5, color="k", lw=0.4, ls=":")
        ax.set_title(f"{SHORT[t]} ({fs[t]['prefix_neg']:,} in prefix)", fontsize=6.5, pad=2)
        ax.set_xticks(x, ks)
    for ax in axes[1]:
        ax.set_xlabel("negative target labels $k$", labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", labelpad=1)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.985), fontsize=5.8,
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.5)
    save(fig, out)


if __name__ == "__main__":
    main()
