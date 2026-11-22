"""Core of the cross-platform trust-transfer study: data, access contract,
features, models and paired statistics.

Access contract (one rule for every method). Each graph is sorted by its
time/order field and cut into equal windows. A *view* of a graph is the triple
(score edges, observable topology, signed edges whose signs may be read):

  target test view   score [0.2,0.4)  topology [0,0.4)  signs [0,0.2)
  target fit view    score [0.1,0.2)  topology [0,0.2)  signs [0,0.1)
  source fit view    score [h/2,h)    topology [0,h)    signs [0,h/2)

No view reads the sign of an edge it scores (tested in test_tt.py).
Tier 1 features read no signs, Tier 2 features read the view's signed edges,
and the Tier 3 (leaked) variant adds the scored edges' own signs.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("TRUSTTRANSFER_DATA", os.path.join(ROOT, "data"))
RESULTS = os.path.join(ROOT, "results")
GRAPHS = ["otc", "alpha", "epinions", "slashdot", "wikirfa", "reddit"]
ECOSYSTEM = {"otc": "crypto", "alpha": "crypto", "epinions": "reviews",
             "slashdot": "news", "wikirfa": "wiki", "reddit": "reddit"}
FAMILIES = {"deg": 4, "topo": 4, "rep": 7}


@dataclass(frozen=True)
class Graph:
    name: str
    s: np.ndarray   # rater ids
    d: np.ndarray   # receiver ids
    y: np.ndarray   # 1 = positive (rating > 0)
    t: np.ndarray   # time or file-order index, sorted ascending
    n: int

    @property
    def m(self):
        return len(self.s)

    def win(self, a, b):
        return np.arange(int(a * self.m), int(b * self.m))


def raw(name):
    """Edge list of one network: columns s (rater), d (receiver), r (rating), t (time/order)."""
    path = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python prepare_data.py` first")
    df = pd.read_csv(path)
    df.columns = ["s", "d", "r", "t"]
    return df


def load(name, ids="sorted", seed=0, df=None):
    """ids: 'sorted' (rank of the original id, the usual loader convention),
    'first_seen' (order of first appearance) or 'random'."""
    df = (raw(name) if df is None else df).sort_values("t", kind="mergesort").reset_index(drop=True)
    nodes = np.unique(np.concatenate([df["s"].to_numpy(), df["d"].to_numpy()]))
    s = np.searchsorted(nodes, df["s"].to_numpy())
    d = np.searchsorted(nodes, df["d"].to_numpy())
    n = len(nodes)
    if ids == "first_seen":
        first = np.full(n, len(df))
        np.minimum.at(first, s, np.arange(len(df)))
        np.minimum.at(first, d, np.arange(len(df)))
        perm = np.empty(n, dtype=np.int64)
        perm[np.argsort(first, kind="stable")] = np.arange(n)
        s, d = perm[s], perm[d]
    elif ids == "random":
        perm = np.random.default_rng(seed).permutation(n)
        s, d = perm[s], perm[d]
    y = (df["r"].to_numpy() > 0).astype(np.int8)
    return Graph(name, s.astype(np.int64), d.astype(np.int64), y, df["t"].to_numpy(), n)


def subgraph(g, keep_nodes, name):
    """Edges with both endpoints in keep_nodes, ids compacted, order kept."""
    mask = np.zeros(g.n, bool)
    mask[keep_nodes] = True
    e = mask[g.s] & mask[g.d]
    remap = np.full(g.n, -1)
    remap[np.flatnonzero(mask)] = np.arange(mask.sum())
    return Graph(name, remap[g.s[e]], remap[g.d[e]], g.y[e], g.t[e], int(mask.sum()))


# ---------------------------------------------------------------- views
@dataclass(frozen=True)
class View:
    g: Graph
    score: np.ndarray
    topo: np.ndarray
    signs: np.ndarray


def target_view(g, signs=None):
    return View(g, g.win(0.2, 0.4), g.win(0.0, 0.4), g.win(0.0, 0.2) if signs is None else signs)


def target_fit_view(g, signs=None):
    """Target-only learner: fit on the later half of the revealed prefix signs."""
    if signs is None:
        return View(g, g.win(0.1, 0.2), g.win(0.0, 0.2), g.win(0.0, 0.1))
    half = len(signs) // 2
    return View(g, signs[half:], g.win(0.0, 0.2), signs[:half])


def source_view(g, h=1.0):
    return View(g, g.win(h / 2, h), g.win(0.0, h), g.win(0.0, h / 2))


# ---------------------------------------------------------------- features
def _undirected(g, idx):
    a = sp.coo_matrix((np.ones(len(idx)), (g.s[idx], g.d[idx])), shape=(g.n, g.n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def node_sign_stats(g, idx):
    s, d, y = g.s[idx], g.d[idx], g.y[idx].astype(np.float64)
    po = np.bincount(s, weights=y, minlength=g.n)
    no = np.bincount(s, minlength=g.n) - po
    pi = np.bincount(d, weights=y, minlength=g.n)
    ni = np.bincount(d, minlength=g.n) - pi
    rec = sp.csr_matrix((2 * y - 1, (s, d)), shape=(g.n, g.n))   # summed signs u->v
    return po, no, pi, ni, rec


def features(view, fams=("deg", "topo", "rep"), leak=False):
    g, idx = view.g, view.score
    u, v = g.s[idx], g.d[idx]
    cols = []
    if "deg" in fams:
        out = np.bincount(g.s[view.topo], minlength=g.n)
        inn = np.bincount(g.d[view.topo], minlength=g.n)
        cols += [np.log1p(out[u]), np.log1p(inn[u]), np.log1p(out[v]), np.log1p(inn[v])]
    if "topo" in fams:
        a = _undirected(g, view.topo)
        deg = np.asarray(a.sum(1)).ravel()
        inv_log = 1.0 / np.log(np.maximum(deg, 2))
        cn, aa = np.empty(len(u)), np.empty(len(u))
        for c in range(0, len(u), 50000):   # chunked: hub rows are wide
            common = a[u[c:c + 50000]].multiply(a[v[c:c + 50000]]).tocsr()
            cn[c:c + 50000] = np.asarray(common.sum(1)).ravel()
            aa[c:c + 50000] = common @ inv_log
        jac = cn / np.maximum(deg[u] + deg[v] - cn, 1)
        cols += [np.log1p(cn), jac, np.log1p(aa), np.log1p(deg[u] * deg[v])]
    if "rep" in fams:
        signs = np.union1d(view.signs, idx) if leak else view.signs
        po, no, pi, ni, rec = node_sign_stats(g, signs)
        recip = np.sign(np.asarray(rec[v, u]).ravel())
        cols += [np.log1p(po[u]), np.log1p(no[u]), (po[u] + 1) / (po[u] + no[u] + 2),
                 np.log1p(pi[v]), np.log1p(ni[v]), (pi[v] + 1) / (pi[v] + ni[v] + 2), recip]
    return np.column_stack(cols).astype(np.float32)


def zscore(x):
    """Per-domain standardisation from the domain's own unlabeled feature values."""
    mu, sd = x.mean(0), x.std(0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def family_cols(fams):
    out, start = [], 0
    for f, k in FAMILIES.items():
        if f in fams:
            out += list(range(start, start + k))
        start += k
    return out


# ---------------------------------------------------------------- metrics
def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def weighted_auc(score, y, w):
    """AUC under per-edge weights w of shape (B, m); ties count one half."""
    order = np.argsort(score, kind="mergesort")
    s, yy, w = score[order], y[order].astype(np.float32), w[:, order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    gp = np.add.reduceat(w * yy, start, axis=1)
    gn = np.add.reduceat(w * (1 - yy), start, axis=1)
    below = np.cumsum(gn, axis=1) - gn
    return (gp * (below + 0.5 * gn)).sum(1) / (gp.sum(1) * gn.sum(1))


def cluster_weights(clusters, b, rng):
    """Bootstrap weights resampling clusters (raters) with replacement."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    c = len(uniq)
    counts = np.stack([np.bincount(rng.integers(0, c, c), minlength=c) for _ in range(b)])
    return counts[:, inv].astype(np.float32)


def paired_delta(score_a, score_b, y, clusters, b=400, seed=0, chunk=50):
    """AUC(a) - AUC(b) on the same edges, rater-cluster bootstrap CI."""
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, b, chunk):
        w = cluster_weights(clusters, min(chunk, b - start), rng)
        deltas.append(weighted_auc(score_a, y, w) - weighted_auc(score_b, y, w))
    deltas = np.concatenate(deltas)
    point = auc(score_a, y) - auc(score_b, y)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi)}


def distrust_precision(score, y, frac, seed=0):
    """Precision on the negative class among the frac lowest-trust edges."""
    k = max(1, int(np.ceil(frac * len(y))))
    jitter = np.random.default_rng(seed).random(len(y)) * 1e-9
    top = np.argsort(score + jitter, kind="mergesort")[:k]
    return float((y[top] == 0).mean())


# ---------------------------------------------------------------- models
def fit_predict(model, xs, ys, xt, seed=0):
    if model == "lr":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        return clf.fit(xs, ys).decision_function(xt)
    if model == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed)
        return clf.fit(xs, ys).predict_proba(xt)[:, 1]
    if model == "coral":
        return fit_predict("lr", coral(xs, xt), ys, xt)
    raise ValueError(model)


def coral(xs, xt, eps=1.0):
    """Recolour source features with the target covariance (Sun et al. 2016)."""
    def msqrt(c, inv=False):
        w, q = np.linalg.eigh(c)
        w = np.maximum(w, 1e-8) ** (-0.5 if inv else 0.5)
        return (q * w) @ q.T
    cs = np.cov(xs, rowvar=False) + eps * np.eye(xs.shape[1])
    ct = np.cov(xt, rowvar=False) + eps * np.eye(xs.shape[1])
    return (xs - xs.mean(0)) @ msqrt(cs, inv=True) @ msqrt(ct) + xt.mean(0)
