"""Dataset table and node-id provenance audit (reviewer R3.6, R4.3).

For every graph: size, positive rate, negatives available to each view, and
whether the node index or the order field encodes account age."""
import json
import os

import numpy as np
from scipy.stats import spearmanr

import tt


def first_seen(g):
    first = np.full(g.n, g.m)
    np.minimum.at(first, g.s, np.arange(g.m))
    np.minimum.at(first, g.d, np.arange(g.m))
    return first


def stratified_auc(score, y, strata):
    """Edge-weighted mean of within-stratum AUCs (strata with one class skipped)."""
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        if 0 < y[m].sum() < m.sum():
            num += tt.auc(score[m], y[m]) * m.sum()
            den += m.sum()
    return num / den


def deciles(x):
    return np.minimum((np.argsort(np.argsort(x, kind="stable")) * 10) // len(x), 9)


def audit(name):
    g = tt.load(name)
    raw = tt.raw(name).sort_values("t", kind="mergesort")
    test = tt.target_view(g).score
    y = g.y[test].astype(int)
    u, v = g.s[test], g.d[test]
    first = first_seen(g)
    row = {"graph": name, "nodes": g.n, "edges": g.m, "pos_rate": float(g.y.mean()),
           "neg_target_prefix": int((g.y[g.win(0, 0.2)] == 0).sum()),
           "neg_source_fit_full": int((g.y[g.win(0.5, 1.0)] == 0).sum()),
           "test_pos_rate": float(y.mean())}
    order_by_rater = raw["s"].to_numpy()
    if np.issubdtype(order_by_rater.dtype, np.number):
        row["rho_order_vs_rater_id"] = float(spearmanr(np.arange(len(raw)), order_by_rater)[0])
    row["rho_sorted_id_vs_first_seen"] = float(spearmanr(np.arange(g.n), first)[0])
    for ids in ("sorted", "first_seen", "random"):
        h = tt.load(name, ids=ids, seed=0)
        prior = -(h.s[test] + h.d[test]).astype(float)
        row[f"id_prior_auc_{ids}"] = tt.auc(prior, y)
    # label shift and cold start between the prefix and the test window (second-round review A.3, A.4)
    pre = g.win(0, 0.2)
    active = np.zeros(g.n, bool)
    active[g.s[pre]] = active[g.d[pre]] = True
    rated = np.zeros(g.n, bool)
    rated[g.s[pre]] = True
    row.update({"neg_rate_prefix": float(1 - g.y[pre].mean()), "neg_rate_fit": float(1 - g.y[g.win(0.1, 0.2)].mean()),
                "neg_rate_test": float(1 - y.mean()), "neg_fit_window": int((g.y[g.win(0.1, 0.2)] == 0).sum()),
                "test_rater_active_in_prefix": float(active[u].mean()),
                "test_receiver_active_in_prefix": float(active[v].mean()),
                "test_rater_rated_in_prefix": float(rated[u].mean())})
    # where does an id signal come from? negative share of received ratings per id decile
    dec = np.minimum(np.arange(g.n) * 10 // g.n, 9)
    neg_in = np.bincount(g.d, weights=1 - g.y, minlength=g.n)
    deg_in = np.bincount(g.d, minlength=g.n)
    row["recv_neg_share_by_id_decile"] = [float(neg_in[dec == k].sum() / max(deg_in[dec == k].sum(), 1))
                                          for k in range(10)]
    top = (dec == 9) & (deg_in > 0)
    first_t = np.full(g.n, np.inf)
    np.minimum.at(first_t, g.d, g.t.astype(float))
    row["top_id_decile_first_received_min_t"] = float(first_t[top].min())
    row["top_id_decile_raw_id_min"] = float(np.unique(np.r_[raw["s"].to_numpy(), raw["d"].to_numpy()])[dec == 9].min()) \
        if np.issubdtype(raw["s"].to_numpy().dtype, np.number) else None
    prior = -(u + v).astype(float)
    row["id_prior_auc_within_receiver_age"] = stratified_auc(prior, y, deciles(first[v]))
    indeg = np.bincount(g.d[g.win(0, 0.4)], minlength=g.n)
    row["id_prior_auc_within_age_x_degree"] = stratified_auc(
        prior, y, deciles(first[v]) * 10 + deciles(indeg[v]))
    return row


if __name__ == "__main__":
    rows = [audit(n) for n in tt.GRAPHS]
    os.makedirs(tt.RESULTS, exist_ok=True)
    with open(os.path.join(tt.RESULTS, "data_audit.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
