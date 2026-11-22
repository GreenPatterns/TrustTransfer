"""Positive control (reviewer R2.4, R2.5): split one platform into two
node-disjoint halves and transfer between them. Identities are disjoint, as in
the cross-platform setting, but the platform is shared, so a protocol that
can detect transfer must show it here. Argument h: source history; h=0.4
gives the source half exactly the target half's window (score [0.2,0.4),
signs [0,0.2)). Writes results/control.json or control_h{h}.json."""
import json
import os
import sys

import numpy as np

import exp_transfer as E
import tt

SEEDS = (0, 1, 2)


def run(name, h):
    g = tt.load(name)
    rows = []
    for sd in SEEDS:
        half = np.random.default_rng(sd).permutation(g.n)[: g.n // 2]
        rest = np.setdiff1d(np.arange(g.n), half)
        a, b = tt.subgraph(g, half, "A"), tt.subgraph(g, rest, "B")
        if E.STRICT:
            sv, tv, fv = tt.source_view_strict(a, h), tt.target_view_strict(b), tt.target_fit_view_strict(b)
        else:
            sv, tv, fv = tt.source_view(a, h), tt.target_view(b), tt.target_fit_view(b)
        xs, xt, xf = tt.features(sv), tt.features(tv), tt.features(fv)
        ys, yt, yf = a.y[sv.score], b.y[tv.score], b.y[fv.score]
        zt = tt.zscore_ref(xt, xf) if E.STRICT else E.zs(xt, E.ALL)
        t_lr = tt.fit_predict("lr", E.zs(xf, E.ALL), yf, zt)
        row = {"edges_A": a.m, "edges_B": b.m, "T-LR": tt.auc(t_lr, yt)}
        for model in ("lr", "gbdt"):
            sc = tt.fit_predict(model, E.zs(xs, E.ALL), ys, zt)
            row[model] = tt.paired_delta(sc, t_lr, yt, b.s[tv.score], b=200, seed=sd)
            row[model]["auc"] = tt.auc(sc, yt)
        rows.append(row)
        print(name, sd, {k: (round(v["delta"], 3) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
    return rows


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    res = {n: run(n, h) for n in tt.GRAPHS}
    out = "control.json" if h == 1.0 else f"control_h{h}.json"
    os.makedirs(E.OUT, exist_ok=True)
    json.dump(res, open(os.path.join(E.OUT, out), "w"), indent=1)
