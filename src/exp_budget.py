"""Label-budget sweep (reviewer R4: when can transfer help?). A fraction q of
the target prefix signs is revealed at random; transfer and target-only
learners see exactly those signs. Writes results/budget.json."""
import json
import os

import numpy as np

import exp_transfer as E
import tt

QS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = (0, 1, 2)


def run(t, models):
    g = tt.load(t)
    prefix = g.win(0.0, 0.2)
    test = tt.target_view(g)
    y = g.y[test.score]
    out = {}
    for q in QS:
        rows = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            rev = np.sort(rng.choice(prefix, max(4, int(q * len(prefix))), replace=False))
            xt = tt.features(tt.target_view(g, signs=rev))
            fit = tt.target_fit_view(g, signs=rev)
            xf, yf = tt.features(fit), g.y[fit.score]
            row = {"T-rep": tt.auc(xt[:, 13], y), "n_neg_revealed": int((g.y[rev] == 0).sum())}
            row["T-LR"] = (tt.auc(tt.fit_predict("lr", E.zs(xf, E.ALL), yf, E.zs(xt, E.ALL)), y)
                           if 0 < yf.sum() < len(yf) else row["T-rep"])
            xfer = [tt.auc(clf.decision_function(E.zs(xt, E.ALL)), y) for s, clf in models.items() if s != t]
            row["transfer_mean"], row["transfer_max"] = float(np.mean(xfer)), float(np.max(xfer))
            rows.append(row)
        out[str(q)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(t, q, {k: round(v, 3) for k, v in out[str(q)].items()}, flush=True)
    return out


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    models = {s: LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        E.zs(E.cached(s)["src1.0_x"], E.ALL), E.cached(s)["src1.0_y"]) for s in tt.GRAPHS}
    res = {t: run(t, models) for t in tt.GRAPHS}
    json.dump(res, open(os.path.join(tt.RESULTS, "budget.json"), "w"), indent=1)
