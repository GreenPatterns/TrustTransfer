"""Fig. 2: the four leakage checks. (a) receiver-reputation rule with and
without the scored edges' signs; (b) transferred reputation block, clean vs
leaky source vs leaky target features, mean over the five sources of each
target; (c) node-index prior under three numberings and within age deciles;
(d) target-only LR when test ratings may read later unsigned ratings (batch)
or only earlier ones (forecast)."""
import numpy as np

from plot_style import C_LEAK, C_REF, SHORT, WIDTH, load, panel, plt, save
import tt


def main():
    targets, pairs, data = load("targets.json"), load("transfer.json"), load("data_audit.json")
    strict = load("strict/targets.json")
    g = tt.GRAPHS
    x = np.arange(len(g))
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.5), gridspec_kw={"width_ratios": [1, 1.12, 1.3, 1]})

    ax = axes[0]
    corr = [targets[t]["T-rep"]["auc"] for t in g]
    leak = [targets[t]["T-rep-leaked"]["auc"] for t in g]
    ax.bar(x - 0.2, corr, 0.4, color="#bbbbbb", label="prefix signs")
    ax.bar(x + 0.2, leak, 0.4, color=C_LEAK, label="+ scored signs")
    panel(ax, "a", "tr")
    ax.set_ylabel("AUC")

    ax = axes[1]
    keys = [("rep", "#bbbbbb", "clean"), ("rep_train_leak", "#9467bd", "src leak"),
            ("rep_test_leak", C_LEAK, "tgt leak")]
    for k, (key, col, lab) in enumerate(keys):
        vals = [np.mean([pairs[f"{s}->{t}"]["audit"][key] for s in g if s != t]) for t in g]
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=col, label=lab)
    panel(ax, "b", "tr")

    ax = axes[2]
    rows = {r["graph"]: r for r in data}
    keys = [("id_prior_auc_sorted", C_REF, "sorted id"), ("id_prior_auc_first_seen", "#8c8c8c", "first seen"),
            ("id_prior_auc_random", "#d9d9d9", "random"), ("id_prior_auc_within_receiver_age", "#2ca02c", "within age")]
    for k, (key, col, lab) in enumerate(keys):
        ax.bar(x + (k - 1.5) * 0.2, [rows[t][key] for t in g], 0.2, color=col, label=lab)
    panel(ax, "c", "tr")

    ax = axes[3]
    ax.bar(x - 0.2, [targets[t]["T-LR"]["auc"] for t in g], 0.4, color=C_LEAK, label="batch")
    ax.bar(x + 0.2, [strict[t]["T-LR"]["auc"] for t in g], 0.4, color="#bbbbbb", label="forecast")
    panel(ax, "d", "tr")

    for ax in axes:
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_ylim(0.25 if ax is axes[2] else 0.4, 1.0)
        ax.set_xticks(x, [SHORT[t] for t in g], rotation=45, ha="right")
        n = len(ax.get_legend_handles_labels()[0])
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  handlelength=0.9, columnspacing=0.6, handletextpad=0.3, fontsize=5.6, labelspacing=0.15,
                  borderaxespad=0.1)
    fig.tight_layout(pad=0.2, w_pad=0.5)
    save(fig, "audits")


if __name__ == "__main__":
    main()
