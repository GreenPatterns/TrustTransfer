"""Transfer AUC by feature group (activity, neighbourhood, reputation, all) for
every source->target pair, plus the target-only model on the same group, in the
style of a per-family transfer table. Logistic regression, full source history.
Writes results/families.json."""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import exp_transfer as E
import tt

GROUPS = {"activity": ("deg",), "neighbourhood": ("topo",), "reputation": ("rep",), "all": ("deg", "topo", "rep")}


def fit(x, y):
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)


def main():
    cols = {g: tt.family_cols(f) for g, f in GROUPS.items()}
    models = {(s, g): fit(E.zs(E.cached(s)["src1.0_x"], c), E.cached(s)["src1.0_y"])
              for s in tt.GRAPHS for g, c in cols.items()}
    out = {"pairs": {}, "target_only": {}}
    for t in tt.GRAPHS:
        ct = E.cached(t)
        y = ct["test_y"]
        out["target_only"][t] = {g: tt.auc(fit(E.zs(ct["tfit_x"], c), ct["tfit_y"]).decision_function(E.zt(ct, c)), y)
                                 for g, c in cols.items()}
        for s in tt.GRAPHS:
            if s != t:
                out["pairs"][f"{s}->{t}"] = {g: tt.auc(models[(s, g)].decision_function(E.zt(ct, c)), y)
                                             for g, c in cols.items()}
        print(t, {g: round(np.mean([out["pairs"][f"{s}->{t}"][g] for s in tt.GRAPHS if s != t]), 3) for g in cols},
              "target-only", {g: round(v, 3) for g, v in out["target_only"][t].items()}, flush=True)
    os.makedirs(E.OUT, exist_ok=True)
    json.dump(out, open(os.path.join(E.OUT, "families.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
