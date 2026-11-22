"""Inductive relational GNN transfer, with and without Deep Graph Infomax
pretraining on the source's own unlabeled graph (single-source, as every method
in the main comparison), a signed GCN (SGCN), plus the target-only models as
in-domain references. Writes gnn.json (batch) or strict/gnn.json; entries of models
not run are kept. Argument: comma-separated models (default GNN,GNN-DGI) and an
output name."""
import json
import os
import sys

import numpy as np

import deep
import exp_transfer as E
import tt

SEEDS = (0, 1, 2)
KIND = {"GNN": "rgnn", "GNN-DGI": "rgnn", "SGCN": "sgcn"}


def inputs(g, c):
    if not E.STRICT:
        return {"test": deep.graph_inputs(tt.target_view(g)), "tfit": deep.graph_inputs(tt.target_fit_view(g)),
                "src": deep.graph_inputs(tt.source_view(g, 1.0))}
    return {"test": deep.snapshots(tt.target_view_strict(g), E.zt(c, E.ALL)),
            "tfit": deep.snapshots(tt.target_fit_view_strict(g), E.zs(c["tfit_x"], E.ALL)),
            "src": deep.snapshots(tt.source_view_strict(g, 1.0), E.zs(c["src1.0_x"], E.ALL))}


def main(models, out_name):
    gi = {n: inputs(tt.load(n), E.cached(n)) for n in tt.GRAPHS}
    path = os.path.join(E.OUT, out_name)
    out = json.load(open(path)) if os.path.exists(path) else {"pairs": {}, "targets": {}}
    t_lr = {}
    for t in tt.GRAPHS:
        c = E.cached(t)
        t_lr[t] = tt.fit_predict("lr", E.zs(c["tfit_x"], E.ALL), c["tfit_y"], E.zt(c, E.ALL))
        for name in models:
            if name == "GNN-DGI":
                continue
            own = E.ensemble([deep.gnn(gi[t]["tfit"], [gi[t]["test"]], sd, kind=KIND[name])[0] for sd in SEEDS])
            key = "T-" + name
            out["targets"].setdefault(t, {})[key] = E.summarise(own, c["test_y"], t_lr[t], c["test_r"])
            print(t, key, round(out["targets"][t][key]["auc"], 3), flush=True)
    for s in tt.GRAPHS:
        targets = [t for t in tt.GRAPHS if t != s]
        tests = [gi[t]["test"] for t in targets]
        for name in models:
            # DGI reads only the source's unlabeled training graph(s): no target or third-party data
            pre = deep._parts(gi[s]["src"]) if name == "GNN-DGI" else None
            runs = [deep.gnn(gi[s]["src"], tests, sd, pretrain=pre, kind=KIND[name]) for sd in SEEDS]
            for k, t in enumerate(targets):
                c = E.cached(t)
                score = E.ensemble([r[k] for r in runs])
                out["pairs"].setdefault(f"{s}->{t}", {})[name] = E.summarise(score, c["test_y"], t_lr[t], c["test_r"])
            print(s, name, {t: round(out["pairs"][f"{s}->{t}"][name]["auc"], 3) for t in targets}, flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(E.OUT, exist_ok=True)
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["GNN", "GNN-DGI"],
         sys.argv[2] if len(sys.argv) > 2 else "gnn.json")
