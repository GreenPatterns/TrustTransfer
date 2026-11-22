"""Construction-choice audit (Sec. 4, Check 2, last sentence).

The reputation features can be built in several plausible ways. This script
varies three choices and measures transferred AUC on all 30 pairs with a 20%
source window and plain logistic regression: (i) receiver statistics counted
from ratings the receiver GAVE rather than RECEIVED, (ii) source training
statistics that include each training edge's own sign, and (iii) raw counts
standardised with source statistics only rather than log counts standardised
per platform. Writes results/construction.json."""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import tt


def signed_degree(g, score, signs, receiver_in):
    po, no, pi, ni, _ = tt.node_sign_stats(g, signs)
    u, v = g.s[score], g.d[score]
    pv, nv = (pi[v], ni[v]) if receiver_in else (po[v], no[v])
    return np.column_stack([po[u], no[u], pv, nv, po[u] / (po[u] + no[u] + 1e-9), pv / (pv + nv + 1e-9)])


def run_pair(gs, gt, receiver_in, own_label, per_domain):
    if own_label:            # statistics over the training window itself
        score = gs.win(0.0, 0.2)
        signs = score
    else:                    # statistics strictly earlier than the scored edges
        score, signs = gs.win(0.1, 0.2), gs.win(0.0, 0.1)
    xs = signed_degree(gs, score, signs, receiver_in)
    tv = tt.target_view(gt)
    xt = signed_degree(gt, tv.score, tv.signs, receiver_in)
    if per_domain:           # log counts, each platform standardised on its own
        xs, xt = (tt.zscore(np.column_stack([np.log1p(x[:, :4]), x[:, 4:]])) for x in (xs, xt))
    else:
        sc = StandardScaler().fit(xs)
        xs, xt = sc.transform(xs), sc.transform(xt)
    clf = LogisticRegression(max_iter=1000).fit(xs, gs.y[score])
    return tt.auc(clf.decision_function(xt), gt.y[tv.score])


if __name__ == "__main__":
    gs = {n: tt.load(n) for n in tt.GRAPHS}
    res = {}
    for per_domain in (False, True):
        for receiver_in in (False, True):
            for own_label in (True, False):
                key = (f"receiver_{'received' if receiver_in else 'given'}__own_label_{own_label}"
                       f"__{'per_domain_log' if per_domain else 'source_scaler_raw'}")
                res[key] = {f"{s}->{t}": run_pair(gs[s], gs[t], receiver_in, own_label, per_domain)
                            for s in tt.GRAPHS for t in tt.GRAPHS if s != t}
                vals = list(res[key].values())
                orig = [res[key][p] for p in ("otc->alpha", "alpha->otc", "otc->epinions")]
                print(key, "mean30", round(np.mean(vals), 3), "orig3", [round(v, 3) for v in orig], flush=True)
    json.dump(res, open(os.path.join(tt.RESULTS, "construction.json"), "w"), indent=1)
