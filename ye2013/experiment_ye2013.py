#!/usr/bin/env python3
"""Ye et al. (WWW 2013) cross-network sign transfer, run under the study's access contract.

Reimplementation of "Predicting Positive and Negative Links in Signed Social
Networks by Transfer Learning" (Ye, Cheng, Zhu, Chen, WWW 2013), the closest
prior work to our setting: it transfers edge-sign knowledge to a newly formed
target network with no shared identities.

Their construction is deliberately sign-free. Sec. 4.1 of that paper states
that no edge sign information enters the features, because target signs are
scarce. Their features are therefore 'unsigned' signals in the paper's terms,
which makes their method the sharpest available test of whether a genuinely
observable signal is portable.

Features (Ye et al. Sec. 4):
  explicit  out-degree(u), in-degree(v), betweenness(u), betweenness(v),
            four triad counts f_FF, f_FB, f_BF, f_BB, edge embeddedness
  latent    NMTF factorization A_s ~ U_s Sigma V_s^T, A_t ~ U_t Sigma V_t^T
            with a SHARED Sigma, so both graphs are expressed in one space

Transfer (Ye et al. Sec. 5): TrAdaBoost-style instance weighting. Source
weights only ever decrease; target-prefix weights only ever increase.

We evaluate under our own protocol: the target labels available for training
are the 20% calibration prefix only, never the test window.
"""
import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ye_data import RESULTS_ROOT, load_dataset

SEEDS = (42, 123, 456)
# NMTF covers the CAP highest-degree nodes (default 3000; YE_CAP=30000 is the coverage check)
CAP = int(os.environ.get("YE_CAP", "3000"))
# YE_RANK=total ranks users by ratings given plus received (covers receivers too);
# the default ranks by ratings given (row sums), as in the main runs
RANK = os.environ.get("YE_RANK", "given")
K_LATENT = 20
N_BOOST = 15
BETWEENNESS_SAMPLES = 200  # pivot sample for approximate betweenness


# --------------------------------------------------------------------------
# Explicit topological features (Ye et al. Sec. 4.1) -- no sign information
# --------------------------------------------------------------------------
def approx_betweenness(edges, num_nodes, n_pivots, rng):
    """Brandes betweenness on a sampled pivot set, unweighted, on the
    undirected projection. Ye et al. use exact betweenness; we sample pivots
    because Epinions has 131k nodes. Sampling is a documented approximation."""
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = np.array([len(a) for a in adj])
    cand = np.flatnonzero(deg > 0)
    if len(cand) == 0:
        return np.zeros(num_nodes)
    pivots = rng.choice(cand, size=min(n_pivots, len(cand)), replace=False)

    bc = np.zeros(num_nodes)
    for s in pivots:
        # BFS single-source shortest paths
        S, P = [], [[] for _ in range(num_nodes)]
        sigma = np.zeros(num_nodes)
        sigma[s] = 1.0
        d = np.full(num_nodes, -1, dtype=np.int64)
        d[s] = 0
        Q = [s]
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in adj[v]:
                if d[w] < 0:
                    d[w] = d[v] + 1
                    Q.append(w)
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(num_nodes)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    scale = len(pivots) / max(1.0, num_nodes)
    return bc / max(scale, 1e-12) / max(num_nodes, 1)


def explicit_features(edges, num_nodes, rng):
    """9 sign-free explicit features per directed edge."""
    out_deg = np.bincount(edges[:, 0], minlength=num_nodes).astype(np.float64)
    in_deg = np.bincount(edges[:, 1], minlength=num_nodes).astype(np.float64)

    # neighbor sets, split by direction, for triads and embeddedness
    succ = [set() for _ in range(num_nodes)]
    pred = [set() for _ in range(num_nodes)]
    for u, v in edges:
        succ[u].add(v)
        pred[v].add(u)

    # Pivot budget scales down with graph size: exact betweenness is O(VE) and
    # Epinions has 131k nodes. Documented approximation, reported in the paper.
    n_piv = int(np.clip(4_000_000 // max(num_nodes, 1), 15, BETWEENNESS_SAMPLES))
    bc = approx_betweenness(edges, num_nodes, n_piv, rng)

    n = len(edges)
    feats = np.zeros((n, 9), dtype=np.float64)
    for i in range(n):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        u_all = succ[u] | pred[u]
        v_all = succ[v] | pred[v]
        common = u_all & v_all
        # four triad types: edge u-w forward/backward x edge w-v forward/backward
        fFF = fFB = fBF = fBB = 0
        for w in common:
            u_fwd = w in succ[u]
            u_bwd = w in pred[u]
            v_fwd = v in succ[w]      # w -> v
            v_bwd = w in succ[v]      # v -> w
            if u_fwd and v_fwd: fFF += 1
            if u_fwd and v_bwd: fFB += 1
            if u_bwd and v_fwd: fBF += 1
            if u_bwd and v_bwd: fBB += 1
        feats[i] = [out_deg[u], in_deg[v], bc[u], bc[v],
                    fFF, fFB, fBF, fBB, len(common)]
    return feats


# --------------------------------------------------------------------------
# Latent topological features (Ye et al. Sec. 4.2): shared-Sigma NMTF
# --------------------------------------------------------------------------
def nmtf_shared(As, At, k, rng, iters=60, alpha=0.1):
    """Minimize ||As - Us S Vs^T||^2 + ||At - Ut S Vt^T||^2 + alpha||S||^2
    by nonnegative multiplicative updates, with S shared across both graphs."""
    M, N = As.shape[0], At.shape[0]
    Us = rng.random((M, k)) + 0.1
    Vs = rng.random((M, k)) + 0.1
    Ut = rng.random((N, k)) + 0.1
    Vt = rng.random((N, k)) + 0.1
    S = rng.random((k, k)) + 0.1
    eps = 1e-9

    for _ in range(iters):
        AsV, AtV = As @ Vs, At @ Vt
        Us *= (AsV @ S.T) / (Us @ S @ (Vs.T @ Vs) @ S.T + eps)
        Ut *= (AtV @ S.T) / (Ut @ S @ (Vt.T @ Vt) @ S.T + eps)
        AsU, AtU = As.T @ Us, At.T @ Ut
        Vs *= (AsU @ S) / (Vs @ S.T @ (Us.T @ Us) @ S + eps)
        Vt *= (AtU @ S) / (Vt @ S.T @ (Ut.T @ Ut) @ S + eps)
        num = Us.T @ (As @ Vs) + Ut.T @ (At @ Vt)
        den = (Us.T @ Us) @ S @ (Vs.T @ Vs) + (Ut.T @ Ut) @ S @ (Vt.T @ Vt) + alpha * S + eps
        S *= num / den
        for X in (Us, Vs, Ut, Vt):
            X /= (X.sum(axis=1, keepdims=True) + eps)
    return Us, Vs, Ut, Vt


def activity(a):
    """Ranking score for the users the factorisation covers (see RANK)."""
    given = np.asarray(a.sum(1)).ravel()
    return given + np.asarray(a.sum(0)).ravel() if RANK == "total" else given


def adjacency(edges, num_nodes):
    d = np.ones(len(edges))
    return csr_matrix((d, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))


def latent_edge_features(edges, U, V):
    """Ye et al. use the outgoing latent profile of u and incoming of v."""
    return np.hstack([U[edges[:, 0]], V[edges[:, 1]]])


# --------------------------------------------------------------------------
# TrAdaBoost (Ye et al. Sec. 5.1; Dai et al. 2007)
# --------------------------------------------------------------------------
def tradaboost(Xs, ys, Xt, yt, Xtest, n_rounds, rng):
    """Source weights only decrease; target weights only increase."""
    n, m = len(ys), len(yt)
    if m == 0:
        clf = LogisticRegression(max_iter=500, random_state=42).fit(Xs, ys)
        return clf.predict_proba(Xtest)[:, 1]

    X = np.vstack([Xs, Xt])
    y = np.concatenate([ys, yt])
    w = np.concatenate([np.ones(n) / n, np.ones(m) / m])
    beta = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(n, 2)) / n_rounds))

    scores, alphas = [], []
    for _ in range(n_rounds):
        p = w / w.sum()
        base = DecisionTreeClassifier(max_depth=3, random_state=int(rng.integers(1e6)))
        base.fit(X, y, sample_weight=p)
        pred = base.predict(X)
        err_vec = np.abs(pred - y)
        eps_t = float(np.sum(p[n:] * err_vec[n:]) / max(p[n:].sum(), 1e-12))
        eps_t = min(max(eps_t, 1e-6), 0.499)
        beta_t = eps_t / (1.0 - eps_t)
        w[:n] *= beta ** (err_vec[:n] / 2.0)
        w[n:] *= beta_t ** (-err_vec[n:] / 2.0)
        scores.append(base.predict_proba(Xtest)[:, 1])
        alphas.append(np.log(1.0 / beta_t))

    half = n_rounds // 2  # Dai et al.: use the second half of the rounds
    S = np.array(scores[half:])
    a = np.array(alphas[half:])
    a = a / max(a.sum(), 1e-12)
    return (S * a[:, None]).sum(axis=0)


# --------------------------------------------------------------------------
def run_pair(source, target, seed, use_latent=True):
    rng = np.random.default_rng(seed)
    sds, tds = load_dataset(source), load_dataset(target)
    se = sds.snapshot_indices[1] + 1
    pre = tds.snapshot_indices[1] + 1
    end = tds.snapshot_indices[3] + 1

    s_edges = sds.edges[:se].astype(np.int64)
    s_y = (sds.labels[:se, 1] > 0.5).astype(int)
    t_pre_edges = tds.edges[:pre].astype(np.int64)
    t_pre_y = (tds.labels[:pre, 1] > 0.5).astype(int)
    t_test_edges = tds.edges[pre:end].astype(np.int64)
    t_test_y = (tds.labels[pre:end, 1] > 0.5).astype(int)

    # Explicit features. Target-side structure uses UNLABELED topology of the
    # full evaluation window, an unsigned signal: no signs are read.
    t_struct = tds.edges[:end].astype(np.int64)
    Xs = explicit_features(s_edges, sds.num_nodes, rng)
    Xt_all = explicit_features(t_struct, tds.num_nodes, rng)
    Xt_pre, Xt_test = Xt_all[:pre], Xt_all[pre:end]

    if use_latent:
        As = adjacency(s_edges, sds.num_nodes)
        At = adjacency(t_struct, tds.num_nodes)
        # NMTF on dense blocks is infeasible at Epinions scale; restrict the
        # factorization to the top-degree induced subgraph and back off to
        # zeros for nodes outside it.
        cap = CAP
        s_keep = np.argsort(-activity(As))[:cap]
        t_keep = np.argsort(-activity(At))[:cap]
        Asd = np.asarray(As[s_keep][:, s_keep].todense(), dtype=np.float64)
        Atd = np.asarray(At[t_keep][:, t_keep].todense(), dtype=np.float64)
        Us_s, Vs_s, Ut_s, Vt_s = nmtf_shared(Asd, Atd, K_LATENT, rng)
        Us = np.zeros((sds.num_nodes, K_LATENT)); Us[s_keep] = Us_s
        Vs = np.zeros((sds.num_nodes, K_LATENT)); Vs[s_keep] = Vs_s
        Ut = np.zeros((tds.num_nodes, K_LATENT)); Ut[t_keep] = Ut_s
        Vt = np.zeros((tds.num_nodes, K_LATENT)); Vt[t_keep] = Vt_s
        Xs = np.hstack([Xs, latent_edge_features(s_edges, Us, Vs)])
        Xt_pre = np.hstack([Xt_pre, latent_edge_features(t_pre_edges, Ut, Vt)])
        Xt_test = np.hstack([Xt_test, latent_edge_features(t_test_edges, Ut, Vt)])

    sc = StandardScaler().fit(Xs)  # source-only standardization, as in our protocol
    Xs, Xt_pre, Xt_test = sc.transform(Xs), sc.transform(Xt_pre), sc.transform(Xt_test)

    out = {}
    p = tradaboost(Xs, s_y, Xt_pre, t_pre_y, Xt_test, N_BOOST, rng)
    out["ye_tradaboost"] = p
    lr = LogisticRegression(max_iter=500, random_state=seed).fit(Xs, s_y)
    out["ye_source_only_lr"] = lr.predict_proba(Xt_test)[:, 1]

    res = {}
    for name, pr in out.items():
        pred = (pr >= 0.5).astype(int)
        res[name] = {
            "auc": float(roc_auc_score(t_test_y, pr)),
            "mcc": float(matthews_corrcoef(t_test_y, pred)),
            "f1": float(f1_score(t_test_y, pred, zero_division=0)),
        }
    res["majority_f1"] = float(f1_score(t_test_y, np.ones_like(t_test_y), zero_division=0))
    res["prevalence"] = float(t_test_y.mean())
    return res


PAIRS = [("otc", "alpha"), ("alpha", "otc"), ("otc", "epinions"),
         ("alpha", "epinions"), ("otc", "wikirfa"), ("alpha", "wikirfa")]
if os.environ.get("YE_PAIRS"):   # e.g. YE_PAIRS=otc-epinions,alpha-epinions
    PAIRS = [tuple(p.split("-")) for p in os.environ["YE_PAIRS"].split(",")]
OUT_NAME = ("ye2013_transfer.json" if CAP == 3000 and RANK == "given"
            else f"ye2013_transfer_cap{CAP}{'_total' if RANK == 'total' else ''}.json")

if __name__ == "__main__":
    d = os.path.join(RESULTS_ROOT, "ye2013")
    os.makedirs(d, exist_ok=True)
    all_rows = []
    print(f"{'pair':<20}{'method':<20}{'AUC':>8}{'MCC':>8}{'F1':>8}{'majF1':>8}")
    print("-" * 72)
    for s, t in PAIRS:
        acc = {}
        t0 = time.time()
        for seed in SEEDS:
            r = run_pair(s, t, seed)
            for m in ("ye_tradaboost", "ye_source_only_lr"):
                acc.setdefault(m, []).append(r[m])
            maj, prev = r["majority_f1"], r["prevalence"]
        for m, runs in acc.items():
            row = {
                "source": s, "target": t, "method": m,
                "auc_mean": float(np.mean([x["auc"] for x in runs])),
                "auc_std": float(np.std([x["auc"] for x in runs], ddof=1)),
                "mcc_mean": float(np.mean([x["mcc"] for x in runs])),
                "f1_mean": float(np.mean([x["f1"] for x in runs])),
                "majority_f1": maj, "prevalence": prev,
                "seeds": list(SEEDS), "wall_s": round(time.time() - t0, 1),
            }
            all_rows.append(row)
            print(f"{s+'->'+t:<20}{m:<20}{row['auc_mean']:>8.3f}"
                  f"{row['mcc_mean']:>8.3f}{row['f1_mean']:>8.3f}{maj:>8.3f}")
    with open(os.path.join(d, OUT_NAME), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved: {os.path.join(d, OUT_NAME)}")
