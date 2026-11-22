"""Main study: all 30 directed pairs over six graphs, feature-level transfer
methods against target-only references under one access contract.

Outputs results/transfer.json (per pair) and results/targets.json (per target).
Each transfer score is compared with the target-only learner (T-LR) on the
same test edges by a rater-cluster paired bootstrap (reviewer R3.4, R3.5)."""
import json
import os
import sys

import numpy as np

import deep
import tt

CACHE = os.path.join(tt.RESULTS, "cache")
ALL = tt.family_cols(("deg", "topo", "rep"))
TIER1 = tt.family_cols(("deg", "topo"))
REP = tt.family_cols(("rep",))
HS = (0.2, 0.5, 0.8, 1.0)
SEEDS = (0, 1, 2)
B = 300


def views(g):
    out = {"test": tt.target_view(g), "tfit": tt.target_fit_view(g)}
    out.update({f"src{h}": tt.source_view(g, h) for h in HS})
    return out


def cached(name):
    """Feature matrices for every view of one graph (computed once)."""
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    g = tt.load(name)
    out = {}
    for key, v in views(g).items():
        out[f"{key}_x"] = tt.features(v)
        out[f"{key}_y"] = g.y[v.score].astype(np.int8)
        out[f"{key}_r"] = g.s[v.score]
    out["test_leak_x"] = tt.features(views(g)["test"], fams=("rep",), leak=True)
    out["src1.0_leak_x"] = tt.features(views(g)["src1.0"], fams=("rep",), leak=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def zs(x, cols):
    return tt.zscore(x[:, cols])


def ensemble(scores):
    return np.mean([tt.zscore(s[:, None])[:, 0] for s in scores], 0)


def summarise(score, y, ref, raters, seed=0):
    row = {"auc": tt.auc(score, y),
           "p1": tt.distrust_precision(score, y, 0.01), "p5": tt.distrust_precision(score, y, 0.05)}
    if ref is not None:
        row["vs_tlr"] = tt.paired_delta(score, ref, y, raters, b=B, seed=seed)
    return row


def target_refs(t):
    c = cached(t)
    x, y, r = c["test_x"], c["test_y"], c["test_r"]
    t_lr = tt.fit_predict("lr", zs(c["tfit_x"], ALL), c["tfit_y"], zs(x, ALL))
    t_gbdt = tt.fit_predict("gbdt", zs(c["tfit_x"], ALL), c["tfit_y"], zs(x, ALL))
    refs = {"T-deg": x[:, 3], "T-rep": x[:, 13], "T-LR": t_lr, "T-GBDT": t_gbdt,
            "T-rep-leaked": c["test_leak_x"][:, 5]}
    rows = {k: summarise(s, y, None, r) for k, s in refs.items()}
    rows["T-GBDT"]["vs_tlr"] = tt.paired_delta(t_gbdt, t_lr, y, r, b=B)
    rows["T-deg"]["vs_tlr"] = tt.paired_delta(x[:, 3], t_lr, y, r, b=B)
    rows["neg_rate"] = float(1 - y.mean())
    return rows, t_lr


_MODELS = {}


def source_model(s, model, key="src1.0", cols=ALL, leak=False):
    """Source-only fits do not depend on the target: fit once, reuse for all targets."""
    tag = (s, model, key, tuple(cols), leak)
    if tag not in _MODELS:
        c = cached(s)
        x = tt.zscore(c["src1.0_leak_x"]) if leak else zs(c[f"{key}_x"], cols)
        if model == "lr":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, c[f"{key}_y"])
            _MODELS[tag] = clf.decision_function
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                 random_state=0).fit(x, c[f"{key}_y"])
            _MODELS[tag] = lambda z, clf=clf: clf.predict_proba(z)[:, 1]
    return _MODELS[tag]


def pair_run(s, t, t_lr):
    cs, ct = cached(s), cached(t)
    xs, ys = cs["src1.0_x"], cs["src1.0_y"]
    xt, yt, rt = ct["test_x"], ct["test_y"], ct["test_r"]
    sc = {"LR-T1": source_model(s, "lr", cols=TIER1)(zs(xt, TIER1)),
          "LR": source_model(s, "lr")(zs(xt, ALL)),
          "GBDT": source_model(s, "gbdt")(zs(xt, ALL)),
          "CORAL": tt.fit_predict("coral", zs(xs, ALL), ys, zs(xt, ALL)),
          "DANN": ensemble([deep.dann(zs(xs, ALL), ys, zs(xt, ALL), sd) for sd in SEEDS])}
    rows = {k: summarise(v, yt, t_lr, rt) for k, v in sc.items()}
    rows["LR-T1"]["vs_tdeg"] = tt.paired_delta(sc["LR-T1"], xt[:, 3], yt, rt, b=B)
    # leakage audits on the reputation block (LR, full source history)
    rep = source_model(s, "lr", cols=REP)
    rows["audit"] = {
        "rep": tt.auc(rep(zs(xt, REP)), yt),
        "rep_test_leak": tt.auc(rep(tt.zscore(ct["test_leak_x"])), yt),
        "rep_train_leak": tt.auc(source_model(s, "lr", cols=REP, leak=True)(zs(xt, REP)), yt)}
    rows["history"] = {}
    for h in HS:
        sc_h = source_model(s, "lr", key=f"src{h}")(zs(xt, ALL))
        rows["history"][str(h)] = {"auc": tt.auc(sc_h, yt), "delta_tlr": tt.auc(sc_h, yt) - tt.auc(t_lr, yt)}
    return rows


def multi_source(t, t_lr):
    ct = cached(t)
    xt, yt, rt = ct["test_x"], ct["test_y"], ct["test_r"]
    envs = [(zs(cached(s)["src1.0_x"], ALL), cached(s)["src1.0_y"]) for s in tt.GRAPHS if s != t]
    x = np.concatenate([e[0] for e in envs])
    y = np.concatenate([e[1] for e in envs])
    w = np.concatenate([np.full(len(e[1]), 1.0 / len(e[1])) for e in envs])
    from sklearn.linear_model import LogisticRegression
    pooled = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        x, y, sample_weight=w * len(y)).decision_function(zs(xt, ALL))
    vr = ensemble([deep.vrex(envs, zs(xt, ALL), sd) for sd in SEEDS])
    return {"Pooled-LR": summarise(pooled, yt, t_lr, rt), "V-REx": summarise(vr, yt, t_lr, rt)}


if __name__ == "__main__":
    graphs = sys.argv[1:] or tt.GRAPHS
    out_t = os.path.join(tt.RESULTS, "targets.json")
    out_p = os.path.join(tt.RESULTS, "transfer.json")
    targets = json.load(open(out_t)) if os.path.exists(out_t) else {}
    pairs = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for t in graphs:
        refs, t_lr = target_refs(t)
        refs.update(multi_source(t, t_lr))
        targets[t] = refs
        for s in tt.GRAPHS:
            if s != t:
                pairs[f"{s}->{t}"] = pair_run(s, t, t_lr)
                print(s, "->", t, {k: round(v["auc"], 3) for k, v in pairs[f"{s}->{t}"].items()
                                   if isinstance(v, dict) and "auc" in v}, flush=True)
        print(t, {k: round(v["auc"], 3) for k, v in refs.items() if isinstance(v, dict)}, flush=True)
        json.dump(targets, open(out_t, "w"), indent=1)
        json.dump(pairs, open(out_p, "w"), indent=1)
