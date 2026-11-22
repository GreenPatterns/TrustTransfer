"""NMTF coverage of the Ye et al. rebuild (paper Sec. 4.5): share of target test
ratings whose two users are both among the CAP users that the factorisation covers,
selected exactly as in experiment_ye2013.run_pair (ratings given, or given plus
received with YE_RANK=total, in the target's graph up to 40%). Writes
results/ye2013/coverage.json."""
import json
import os

import numpy as np

from experiment_ye2013 import adjacency
from ye_data import RESULTS_ROOT, load_dataset


def coverage(target, cap, rank="given"):
    tds = load_dataset(target)
    pre, end = tds.snapshot_indices[1] + 1, tds.snapshot_indices[3] + 1
    a = adjacency(tds.edges[:end].astype(np.int64), tds.num_nodes)
    score = np.asarray(a.sum(1)).ravel()
    if rank == "total":
        score = score + np.asarray(a.sum(0)).ravel()
    keep = np.zeros(tds.num_nodes, bool)
    keep[np.argsort(-score)[:cap]] = True
    test = tds.edges[pre:end].astype(np.int64)
    both = keep[test[:, 0]] & keep[test[:, 1]]
    return {"target": target, "cap": cap, "rank": rank, "nodes_covered": float(keep.mean()),
            "test_ratings_both_users_covered": float(both.mean())}


if __name__ == "__main__":
    rows = [coverage(t, 3000) for t in ("alpha", "otc", "epinions", "wikirfa")] + \
        [coverage("epinions", 30000), coverage("epinions", 50000, "total")]
    for r in rows:
        print(r)
    with open(os.path.join(RESULTS_ROOT, "ye2013", "coverage.json"), "w") as f:
        json.dump(rows, f, indent=1)
