"""Few-shot transfer at fixed negative-label budgets (second-round reviews A.2, B.2).

For every target and budget k, a draw reveals k negative ratings and the matching
number of positive ratings (the prefix's own positive:negative ratio), sampled from
the target's first 20% of ratings. Every learner sees exactly these labels:

  T-LR      target-only logistic regression (needs a negative and a positive
            training example; otherwise the label-free rule T-Deg is used)
  ZS-LR     source logistic regression, no target labels (zero-shot)
  WS-LR     warm start: target LR whose L2 penalty pulls the weights toward the
            source weights instead of zero (MAP adaptation, Chelba and Acero 2006)
  Pool-LR   one LR on source and target ratings, the two platforms weighted equally
            (a fixed class-stratified sample of POOL_ROWS source ratings, for speed)
  WS-GBDT   source gradient boosting continued with 50 trees on the target labels

Labels in the training window [10%,20%) are training examples; their reputation
features use the revealed labels of [0%,10%) only. Test reputation features use all
revealed labels. Contract (batch or strict) as in exp_transfer (TT_CONTRACT).
Writes fewshot.json in the contract's results folder."""
import copy
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import exp_transfer as E
import tt

KS = (0, 5, 10, 25, 50, 100)
DRAWS = 10
EXTRA_TREES = 50
POOL_ROWS = 20000
DEG_TOPO = list(range(8))


def balanced_weights(y):
    n1, n0 = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def map_lr(x, y, w0, b0, c=1.0):
    """Class-balanced logistic loss + 0.5 * ||w - w0||^2 (sklearn's objective with
    the prior mean moved from zero to w0); the intercept is not penalised."""
    if len(y) == 0:
        return w0, b0
    cw, ys = c * balanced_weights(y), 2.0 * y - 1

    def obj(theta):
        w, b = theta[:-1], theta[-1]
        m = -ys * (x @ w + b)
        g = cw * expit(m) * -ys
        return (cw * np.logaddexp(0, m)).sum() + 0.5 * ((w - w0) ** 2).sum(), np.r_[x.T @ g + (w - w0), g.sum()]

    theta = minimize(obj, np.r_[w0, b0], jac=True, method="L-BFGS-B").x
    return theta[:-1], theta[-1]


def sources():
    """Source-only models on each platform's full training view (fit once)."""
    out = {}
    for s in tt.GRAPHS:
        c = E.cached(s)
        x, y = E.zs(c["src1.0_x"], E.ALL), c["src1.0_y"]
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)
        gb = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(x, y)
        rng = np.random.default_rng(0)
        keep = np.arange(len(y)) if len(y) <= POOL_ROWS else np.concatenate(
            [rng.choice(np.flatnonzero(y == k), int(round(POOL_ROWS * (y == k).mean())), replace=False) for k in (0, 1)])
        out[s] = {"x": x[keep], "y": y[keep], "w": lr.coef_[0], "b": lr.intercept_[0], "gbdt": gb}
    return out


def reveal(g, k, rng):
    """k prefix negatives and round(k * pos/neg) prefix positives, uniformly at random."""
    pre = g.win(0.0, 0.2)
    neg, pos = pre[g.y[pre] == 0], pre[g.y[pre] == 1]
    k = min(k, len(neg))
    n_pos = int(round(k * len(pos) / len(neg)))
    return np.sort(np.r_[rng.choice(neg, k, replace=False), rng.choice(pos, n_pos, replace=False)])


def rep(g, score, signs):
    return tt.features(tt.View(g, score, score[:0], signs), fams=("rep",))


def draw_scores(g, c, revealed, src, t):
    """Scores of every learner on the target test edges for one set of revealed labels."""
    fit_idx = g.win(0.1, 0.2)
    ref = np.c_[c["tfit_x"][:, DEG_TOPO], rep(g, fit_idx, revealed[revealed < fit_idx[0]])]
    xt = np.c_[c["test_x"][:, DEG_TOPO], rep(g, g.win(0.2, 0.4), revealed)]
    xt = tt.zscore_ref(xt, ref) if E.STRICT else tt.zscore(xt)
    train = np.isin(fit_idx, revealed)
    xf, yf = tt.zscore_ref(ref[train], ref), c["tfit_y"][train].astype(float)
    n_neg = int((yf == 0).sum())
    out = {"T-Deg": c["test_x"][:, 3]}
    if 0 < n_neg < len(yf):
        out["T-LR"] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xf, yf).decision_function(xt)
    else:
        out["T-LR"] = out["T-Deg"]
    for s, m in src.items():
        if s == t:
            continue
        out[f"ZS-LR|{s}"] = xt @ m["w"] + m["b"]
        w, b = map_lr(xf, yf, m["w"], m["b"])
        out[f"WS-LR|{s}"] = xt @ w + b
        xp, yp = np.r_[m["x"], xf], np.r_[m["y"], yf]
        dom = np.r_[np.full(len(m["y"]), 0.5 / len(m["y"])), np.full(len(yf), 0.5 / max(len(yf), 1))]
        wp = dom * np.r_[balanced_weights(m["y"]), balanced_weights(yf)] * len(yp)
        out[f"Pool-LR|{s}"] = LogisticRegression(max_iter=2000).fit(xp, yp, sample_weight=wp).decision_function(xt)
        gb = m["gbdt"]
        if 0 < n_neg < len(yf):
            gb = copy.deepcopy(gb).set_params(warm_start=True, early_stopping=False,
                                              max_iter=gb.n_iter_ + EXTRA_TREES)
            gb.fit(xf, yf)
        out[f"WS-GBDT|{s}"] = gb.predict_proba(xt)[:, 1]
    return out, n_neg


def run_target(t, src):
    g, c = tt.load(t), E.cached(t)
    y = c["test_y"]
    pre = g.win(0.0, 0.2)
    res = {"prefix_neg": int((g.y[pre] == 0).sum()), "budgets": {}}
    for k in KS:
        rows = []
        for d in range(DRAWS):
            revealed = reveal(g, k, np.random.default_rng(1000 * k + d))
            scores, n_neg = draw_scores(g, c, revealed, src, t)
            rows.append({"n_revealed": len(revealed), "n_neg_train": n_neg,
                         **{name: tt.auc(sc, y) for name, sc in scores.items()}})
        res["budgets"][str(k)] = rows
        means = {n: np.mean([r[n] for r in rows]) for n in rows[0]}
        summary = {m: round(np.mean([v for n, v in means.items() if n.startswith(m + "|")]), 3)
                   for m in ("ZS-LR", "WS-LR", "Pool-LR", "WS-GBDT")}
        print(t, k, "T-LR %.3f" % means["T-LR"], summary, "neg_train %.1f" % means["n_neg_train"], flush=True)
    return res


def merge():
    """Collect the per-target files into fewshot.json once all six exist."""
    parts = {t: os.path.join(E.OUT, f"fewshot_{t}.json") for t in tt.GRAPHS}
    if all(os.path.exists(p) for p in parts.values()):
        json.dump({t: json.load(open(p)) for t, p in parts.items()},
                  open(os.path.join(E.OUT, "fewshot.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(E.OUT, exist_ok=True)
    src = sources()
    for t in sys.argv[1:] or tt.GRAPHS:
        json.dump(run_target(t, src), open(os.path.join(E.OUT, f"fewshot_{t}.json"), "w"), indent=1)
    merge()
