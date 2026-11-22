"""Fig. 3: every transferred model against the target-only references, per
target. Each dot is one (method, source) pair: 6 methods x 5 sources = 30 per
target; colour marks whether source and target share an interaction type."""
import numpy as np

from plot_style import C_CROSS, C_REF, C_SAME, SAME_TYPE, SHORT, WIDTH, load, plt, save
import tt

METHODS = ["LR", "GBDT", "CORAL", "DANN", "GNN", "GNN-DGI"]


def main():
    pairs, gnn, targets = load("transfer.json"), load("gnn.json"), load("targets.json")
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6))
    rng = np.random.default_rng(0)
    for i, t in enumerate(tt.GRAPHS):
        for s in tt.GRAPHS:
            if s == t:
                continue
            same = (s, t) in SAME_TYPE
            for k, m in enumerate(METHODS):
                src = gnn["pairs"] if m.startswith("GNN") else pairs
                y = i + (k - 2.5) * 0.06 + rng.uniform(-0.02, 0.02)
                ax.plot(src[f"{s}->{t}"][m]["auc"], y, "o", ms=2.6 if same else 2.0, alpha=0.9 if same else 0.55,
                        color=C_SAME if same else C_CROSS, mec="none", zorder=3 if same else 2)
        ref = targets[t]
        ax.plot([ref["T-LR"]["auc"]] * 2, [i - 0.33, i + 0.33], color=C_REF, lw=1.6, zorder=4)
        ax.plot(gnn["targets"][t]["T-GNN"]["auc"], i, "D", ms=3.2, mfc="none", mec=C_REF, mew=0.8, zorder=4)
        ax.plot(ref["T-Rep" if "T-Rep" in ref else "T-rep"]["auc"], i, "^", ms=3, color="#999999", zorder=4)
        ax.plot(ref["T-deg"]["auc"], i, "v", ms=3, color="#999999", zorder=4)
    ax.set_yticks(range(len(tt.GRAPHS)), [SHORT[t] for t in tt.GRAPHS])
    ax.invert_yaxis()
    ax.set_xlabel("test AUC on the target")
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_SAME, ms=3, label="transfer, same interaction type"),
               plt.Line2D([], [], marker="o", ls="", color=C_CROSS, ms=3, label="transfer, other type"),
               plt.Line2D([], [], color=C_REF, lw=1.6, label="T-LR (target-only)"),
               plt.Line2D([], [], marker="D", ls="", mfc="none", mec=C_REF, ms=3, label="T-GNN"),
               plt.Line2D([], [], marker="^", ls="", color="#999999", ms=3, label="T-Rep"),
               plt.Line2D([], [], marker="v", ls="", color="#999999", ms=3, label="T-Deg")]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(pad=0.2)
    save(fig, "targets")


if __name__ == "__main__":
    main()
