"""Turn results/*.json into the numbers behind every table and claim in the
paper, written to results/tables.txt. Figures are drawn by fig_*.py."""
import json
import os

import matplotlib
import numpy as np

import tt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = tt.RESULTS
FIG = os.path.join(tt.ROOT, "figures")
SHORT = {"otc": "OTC", "alpha": "Alpha", "epinions": "Epin.", "slashdot": "Slash.",
         "wikirfa": "RfA", "reddit": "Reddit"}
PAIRS = [(s, t) for s in tt.GRAPHS for t in tt.GRAPHS if s != t]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def domain_boot(values, b=2000, seed=0):
    """Mean over directed pairs; CI by resampling graphs (domains) with replacement."""
    rng = np.random.default_rng(seed)
    idx = {g: i for i, g in enumerate(tt.GRAPHS)}
    mat = np.full((6, 6), np.nan)
    for (s, t), v in values.items():
        mat[idx[s], idx[t]] = v
    means = []
    for _ in range(b):
        pick = rng.integers(0, 6, 6)
        sub = mat[np.ix_(pick, pick)]
        same = pick[:, None] == pick[None, :]
        vals = sub[~same & ~np.isnan(sub)]
        if len(vals):
            means.append(vals.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.nanmean(list(values.values()))), float(lo), float(hi)


def method_rows(pairs, gnn, out):
    out.append("% method | mean AUC | mean dAUC vs T-LR [domain CI] | pairs CI>0 | pairs CI<0")
    names = ["LR-T1", "LR", "GBDT", "CORAL", "DANN"]
    rows = {}
    for m in names + ["GNN", "GNN-DGI"]:
        src = gnn["pairs"] if m.startswith("GNN") else pairs
        if src is None:
            continue
        cells = {(s, t): src[f"{s}->{t}"][m] for s, t in PAIRS if m in src.get(f"{s}->{t}", {})}
        if not cells:
            continue
        d = {k: v["vs_tlr"]["delta"] for k, v in cells.items()}
        mu, lo, hi = domain_boot(d)
        win = sum(v["vs_tlr"]["lo"] > 0 for v in cells.values())
        loss = sum(v["vs_tlr"]["hi"] < 0 for v in cells.values())
        aucm = np.mean([v["auc"] for v in cells.values()])
        rows[m] = (aucm, mu, lo, hi, win, loss)
        out.append(f"{m} & {aucm:.3f} & {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] & {win} & {loss} \\\\")
    return rows


def main():
    out = []
    data = load("data_audit.json")
    targets, pairs, gnn = load("targets.json"), load("transfer.json"), load("gnn.json")
    out.append("% graph | nodes | edges | pos | neg prefix | id-prior sorted / first-seen / random | within-age")
    for r in data:
        out.append(f"{SHORT[r['graph']]} & {r['nodes']:,} & {r['edges']:,} & {r['pos_rate']:.2f} & "
                   f"{r['neg_target_prefix']:,} & {r['id_prior_auc_sorted']:.2f} & "
                   f"{r['id_prior_auc_first_seen']:.2f} & {r['id_prior_auc_random']:.2f} & "
                   f"{r['id_prior_auc_within_receiver_age']:.2f} & rho={r['rho_sorted_id_vs_first_seen']:.2f} \\\\")
    if targets:
        out.append("% target refs: T-deg T-rep T-LR T-GBDT T-GNN Pooled V-REx | T-rep leaked | neg rate")
        for t in tt.GRAPHS:
            if t not in targets:
                continue
            r = targets[t]
            tg = gnn["targets"][t]["T-GNN"]["auc"] if gnn and t in gnn.get("targets", {}) else float("nan")
            out.append(f"{SHORT[t]} & {r['T-deg']['auc']:.3f} & {r['T-rep']['auc']:.3f} & {r['T-LR']['auc']:.3f} & "
                       f"{r['T-GBDT']['auc']:.3f} & {tg:.3f} & {r['Pooled-LR']['auc']:.3f} & {r['V-REx']['auc']:.3f}"
                       f" & leak {r['T-rep-leaked']['auc']:.3f} & neg {r['neg_rate']:.3f}"
                       f" & pooled d {r['Pooled-LR']['vs_tlr']['delta']:+.3f} [{r['Pooled-LR']['vs_tlr']['lo']:+.3f},"
                       f"{r['Pooled-LR']['vs_tlr']['hi']:+.3f}] vrex d {r['V-REx']['vs_tlr']['delta']:+.3f} \\\\")
    if pairs:
        rows = method_rows(pairs, gnn, out)
        d = {tuple(k.split("->")): v["LR-T1"]["vs_tdeg"]["delta"] for k, v in pairs.items()}
        mu, lo, hi = domain_boot(d)
        up = sum(v["LR-T1"]["vs_tdeg"]["lo"] > 0 for v in pairs.values())
        down = sum(v["LR-T1"]["vs_tdeg"]["hi"] < 0 for v in pairs.values())
        out.append(f"% LR-U minus T-Deg: mean {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]; pairs CI>0 {up}, CI<0 {down}")
        a = {k: np.mean([pairs[f"{s}->{t}"]["audit"][k] for s, t in PAIRS if f"{s}->{t}" in pairs])
             for k in ("rep", "rep_test_leak", "rep_train_leak")}
        out.append(f"% audit means over pairs: {a}")
        for h in ("0.2", "0.5", "0.8", "1.0"):
            v = [pairs[f"{s}->{t}"]["history"][h] for s, t in PAIRS if f"{s}->{t}" in pairs]
            out.append(f"% history h={h}: AUC {np.mean([x['auc'] for x in v]):.3f}  "
                       f"d vs T-LR {np.mean([x['delta_tlr'] for x in v]):+.3f}")
        same = [pairs[p]["LR"]["vs_tlr"]["delta"] for p in ("otc->alpha", "alpha->otc") if p in pairs]
        out.append(f"% same-ecosystem LR deltas: {same}")
        for t in tt.GRAPHS:
            got = {s: pairs[f"{s}->{t}"]["LR"] for s in tt.GRAPHS if f"{s}->{t}" in pairs}
            if got and targets and t in targets:
                lift = lambda d: d["p5"] / targets[t]["neg_rate"]
                out.append(f"% P@5% lift {t}: T-LR {lift(targets[t]['T-LR']):.2f} T-deg {lift(targets[t]['T-deg']):.2f} "
                           f"T-rep {lift(targets[t]['T-rep']):.2f} LR mean {np.mean([lift(v) for v in got.values()]):.2f} "
                           f"LR max {max(lift(v) for v in got.values()):.2f}")
    construction = load("construction.json")
    if construction:
        for k, v in construction.items():
            orig = [v[p] for p in ("otc->alpha", "alpha->otc", "otc->epinions")]
            out.append(f"% construction {k}: mean over 30 pairs {np.mean(list(v.values())):.3f}; "
                       f"otc->alpha, alpha->otc, otc->epinions {np.round(orig, 3).tolist()}")
    for fname in ("control.json", "control_h0.4.json"):
        ctrl = load(fname)
        for g, rows in (ctrl or {}).items():
            out.append(f"% {fname} {g}: T-LR {np.mean([r['T-LR'] for r in rows]):.3f} "
                       f"LR {np.mean([r['lr']['auc'] for r in rows]):.3f} "
                       f"d {np.mean([r['lr']['delta'] for r in rows]):+.3f} "
                       f"(CI>0 {sum(r['lr']['lo'] > 0 for r in rows)}/3, CI<0 {sum(r['lr']['hi'] < 0 for r in rows)}/3) "
                       f"GBDT d {np.mean([r['gbdt']['delta'] for r in rows]):+.3f}")
    bud = load("budget.json")
    if bud:
        budget_text(bud, out)
    if targets:
        for k in ("Pooled-LR", "V-REx"):
            out.append(f"% {k}: CI<0 on {sum(targets[t][k]['vs_tlr']['hi'] < 0 for t in tt.GRAPHS)} of 6 targets; "
                       f"mean d {np.mean([targets[t][k]['vs_tlr']['delta'] for t in tt.GRAPHS]):+.3f}")
        triage_rows(targets, pairs, out)
        families_rows(out)
    open(os.path.join(R, "tables.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def budget_text(bud, out):
    qs = sorted(bud[tt.GRAPHS[0]], key=float)
    for t in bud:
        cross = [q for q in qs if bud[t][q]["T-LR"] >= bud[t][q]["transfer_mean"]]
        out.append(f"% budget {t}: first q where T-LR >= mean transfer: {cross[0] if cross else 'none'}; "
                   f"neg revealed at q: {[round(bud[t][q]['n_neg_revealed']) for q in qs]}; "
                   f"T-LR minus best source: {[round(bud[t][q]['T-LR'] - bud[t][q]['transfer_max'], 3) for q in qs]}")


def triage_rows(targets, pairs, out):
    """Table 3: distrust found in the lowest-trust 1% / 5% of test edges, as lift over the base rate."""
    out.append("% triage: target | base | T-Deg | T-Rep | T-LR | LR mean | LR best (source), each P@1%/P@5% lift")
    for t in tt.GRAPHS:
        base = targets[t]["neg_rate"]
        lift = lambda d: (d["p1"] / base, d["p5"] / base)
        got = {s: pairs[f"{s}->{t}"]["LR"] for s in tt.GRAPHS if s != t}
        best = max(got, key=lambda s: got[s]["p5"])
        mean = (np.mean([lift(v)[0] for v in got.values()]), np.mean([lift(v)[1] for v in got.values()]))
        cells = [lift(targets[t]["T-deg"]), lift(targets[t]["T-rep"]), lift(targets[t]["T-LR"]), mean, lift(got[best])]
        out.append(f"{SHORT[t]} & {base:.3f} & " + " & ".join(f"{a:.1f} / {b:.1f}" for a, b in cells)
                   + f" & {SHORT[best]} \\\\")



def families_rows(out):
    """Table 3 bottom block: mean transfer AUC over the five sources / target-only AUC, per feature group."""
    fam = load("families.json")
    if not fam:
        return
    out.append("% families: target | activity | neighbourhood | reputation | all (mean transfer / target-only)")
    for t in tt.GRAPHS:
        cells = []
        for g in ("activity", "neighbourhood", "reputation", "all"):
            m = np.mean([fam["pairs"][f"{s}->{t}"][g] for s in tt.GRAPHS if s != t])
            cells.append(f"{m:.3f}/{fam['target_only'][t][g]:.3f}")
        out.append(f"% families {t}: " + " ".join(cells))


if __name__ == "__main__":
    main()
