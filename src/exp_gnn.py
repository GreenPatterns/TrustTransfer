"""Inductive relational GNN transfer, with and without multi-platform DGI
pretraining (reviewer R1.3, R2.5), plus the target-only GNN as in-domain
reference (R3.3). Writes results/gnn.json."""
import json
import os

import numpy as np

import deep
import exp_transfer as E
import tt

SEEDS = (0, 1, 2)


def main():
    gs = {n: tt.load(n) for n in tt.GRAPHS}
    gi = {n: {"test": deep.graph_inputs(tt.target_view(g)), "tfit": deep.graph_inputs(tt.target_fit_view(g)),
              "src": deep.graph_inputs(tt.source_view(g, 1.0))} for n, g in gs.items()}
    pretrain = [gi[n]["test"] for n in tt.GRAPHS]
    t_lr, out = {}, {"pairs": {}, "targets": {}}
    for t in tt.GRAPHS:
        c = E.cached(t)
        t_lr[t] = tt.fit_predict("lr", E.zs(c["tfit_x"], E.ALL), c["tfit_y"], E.zs(c["test_x"], E.ALL))
        own = E.ensemble([deep.gnn(gi[t]["tfit"], [gi[t]["test"]], sd)[0] for sd in SEEDS])
        out["targets"][t] = {"T-GNN": E.summarise(own, c["test_y"], t_lr[t], c["test_r"])}
        print(t, "T-GNN", round(out["targets"][t]["T-GNN"]["auc"], 3), flush=True)
    for s in tt.GRAPHS:
        targets = [t for t in tt.GRAPHS if t != s]
        tests = [gi[t]["test"] for t in targets]
        for name, pre in (("GNN", None), ("GNN-DGI", pretrain)):
            runs = [deep.gnn(gi[s]["src"], tests, sd, pretrain=pre) for sd in SEEDS]
            for k, t in enumerate(targets):
                c = E.cached(t)
                score = E.ensemble([r[k] for r in runs])
                out["pairs"].setdefault(f"{s}->{t}", {})[name] = E.summarise(score, c["test_y"], t_lr[t], c["test_r"])
            print(s, name, {t: round(out["pairs"][f"{s}->{t}"][name]["auc"], 3) for t in targets}, flush=True)
        json.dump(out, open(os.path.join(tt.RESULTS, "gnn.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
