"""Data adapter for experiment_ye2013.py: presents a network with the fields
that script reads (edges sorted by time, one-hot labels, timestamps,
num_nodes, and the ten equal-size snapshot boundaries)."""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tt  # noqa: E402

RESULTS_ROOT = tt.RESULTS
ALIASES = {"otc": "otc", "alpha": "alpha", "epinions": "epinions", "wikirfa": "wikirfa"}


def snapshot_indices(m, slots=10):
    """Last edge index of each snapshot; the remainder goes to the earliest snapshots."""
    base, extra = divmod(m, slots)
    ends, cur = [], -1
    for i in range(slots):
        cur += base + (1 if i < extra else 0)
        ends.append(cur)
    ends[-1] = m - 1
    return ends


@dataclass(frozen=True)
class Dataset:
    edges: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    num_nodes: int
    snapshot_indices: list


def load_dataset(name):
    g = tt.load(ALIASES[name.lower()])
    labels = np.zeros((g.m, 2), dtype=np.float32)
    labels[np.arange(g.m), g.y.astype(int)] = 1.0
    return Dataset(np.stack([g.s, g.d], axis=1), labels, g.t, g.n, snapshot_indices(g.m))
