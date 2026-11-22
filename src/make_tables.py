"""Turn results/*.json into the paper's numbers (tables.txt). Every number in the
paper comes from this script's output. Two contracts: forecast (strict, the main
one; results/strict/) and batch (results/)."""
import json
import os

import numpy as np

import tt

R = tt.RESULTS
CONTRACTS = {"forecast": os.path.join(R, "strict"), "batch": R}
SHORT = {"otc": "OTC", "alpha": "Alpha", "epinions": "Epin.", "slashdot": "Slash.",
         "wikirfa": "RfA", "reddit": "Reddit"}
PAIRS = [(s, t) for s in tt.GRAPHS for t in tt.GRAPHS if s != t]
TIMESTAMPED, FILE_ORDER = ("otc", "alpha", "wikirfa", "reddit"), ("epinions", "slashdot")
BITCOIN = (("otc", "alpha"), ("alpha", "otc"))
SAME_KIND = BITCOIN + (("epinions", "slashdot"), ("slashdot", "epinions"))
SINGLE = ["LR-T1", "LR", "GBDT", "CORAL", "DANN", "GNN", "GNN-DGI", "SGCN"]
FRACS = (0.01, 0.05)


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def domain_boot(values, b=2000, seed=0):
    """Mean over directed pairs; CI by resampling the six platforms with replacement.
    A resample keeps every pair between two draws of *different* platforms, with
    multiplicity (a platform drawn twice contributes its pairs twice)."""
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


def target_boot(values, b=2000, seed=0):
    """Mean over targets; CI by resampling the six targets with replacement."""
    rng = np.random.default_rng(seed)
    v = np.array([values[t] for t in tt.GRAPHS])
    means = [v[rng.integers(0, 6, 6)].mean() for _ in range(b)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def lopo(values):
    """Mean over pairs after leaving out each platform (as source and as target)."""
    m = [np.mean([v for (s, t), v in values.items() if p not in (s, t)]) for p in tt.GRAPHS]
    return float(min(m)), float(max(m)), tt.GRAPHS[int(np.argmin(m))], tt.GRAPHS[int(np.argmax(m))]


def method_cells(m, c):
    if m == "SGCN" and c["sgcn"] is not None:
        src = c["sgcn"]["pairs"]
    elif m.startswith("GNN") or m == "SGCN":
        src = (c["gnn"] or {}).get("pairs", {})
    else:
        src = c["pairs"] or {}
    return {(s, t): src[f"{s}->{t}"][m] for s, t in PAIRS if m in src.get(f"{s}->{t}", {})}


def method_rows(c, out, tag):
    out.append(f"% [{tag}] method | AUC | dAUC [platform CI] | CI>0 | CI<0 | timestamped d | file-order d | "
               "Bitcoin pair d | same-kind d | other d | LOPO range")
    for m in SINGLE:
        cells = method_cells(m, c)
        if len(cells) < 30:
            continue
        d = {k: v["vs_tlr"]["delta"] for k, v in cells.items()}
        mu, lo, hi = domain_boot(d)
        win = sum(v["vs_tlr"]["lo"] > 0 for v in cells.values())
        loss = sum(v["vs_tlr"]["hi"] < 0 for v in cells.values())
        ts = np.mean([v for (s, t), v in d.items() if t in TIMESTAMPED])
        fo = np.mean([v for (s, t), v in d.items() if t in FILE_ORDER])
        btc = [round(d[p], 3) for p in BITCOIN]
        same = np.mean([d[p] for p in SAME_KIND])
        other = np.mean([v for p, v in d.items() if p not in SAME_KIND])
        l1, l2, p1, p2 = lopo(d)
        out.append(f"{m} & {np.mean([v['auc'] for v in cells.values()]):.3f} & {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] & "
                   f"{win} & {loss} & ts {ts:+.3f} & fo {fo:+.3f} & btc {btc} & same {same:+.3f} & other {other:+.3f}"
                   f" & lopo [{l1:+.3f} w/o {p1}, {l2:+.3f} w/o {p2}] \\\\")
    for m in ("Pooled-LR", "V-REx"):
        d = {t: c["targets"][t][m]["vs_tlr"]["delta"] for t in tt.GRAPHS}
        mu, lo, hi = target_boot(d)
        loss = sum(c["targets"][t][m]["vs_tlr"]["hi"] < 0 for t in tt.GRAPHS)
        win = sum(c["targets"][t][m]["vs_tlr"]["lo"] > 0 for t in tt.GRAPHS)
        out.append(f"{m} & {np.mean([c['targets'][t][m]['auc'] for t in tt.GRAPHS]):.3f} & {mu:+.3f} "
                   f"[{lo:+.3f}, {hi:+.3f}] & {win} & {loss} (of 6 targets) & per target "
                   + " ".join(f"{SHORT[t]} {d[t]:+.3f}" for t in tt.GRAPHS) + " \\\\")


def target_rows(c, out, tag):
    out.append(f"% [{tag}] target-only AUC: target | T-Deg | T-Rep | T-LR | T-GBDT | T-GNN | T-SGCN | neg rate")
    g, s = (c["gnn"] or {}).get("targets", {}), (c["sgcn"] or c["gnn"] or {}).get("targets", {})
    cols = {k: [] for k in ("T-deg", "T-rep", "T-LR", "T-GBDT", "T-GNN", "T-SGCN")}
    for t in tt.GRAPHS:
        r = c["targets"][t]
        vals = {k: r[k]["auc"] for k in ("T-deg", "T-rep", "T-LR", "T-GBDT")}
        vals["T-GNN"] = g.get(t, {}).get("T-GNN", {}).get("auc", np.nan)
        vals["T-SGCN"] = s.get(t, {}).get("T-SGCN", {}).get("auc", np.nan)
        for k, v in vals.items():
            cols[k].append(v)
        out.append(f"{SHORT[t]} & " + " & ".join(f"{v:.3f}" for v in vals.values()) + f" & {r['neg_rate']:.3f} \\\\")
    out.append("% [" + tag + "] means: " + " ".join(f"{k} {np.nanmean(v):.3f}" for k, v in cols.items()))


def pair_table(c, out, tag):
    out.append(f"% [{tag}] transfer AUC (LR, all features); rows sources, columns targets; * = CI>0 vs T-LR")
    for s in tt.GRAPHS:
        cells = []
        for t in tt.GRAPHS:
            if s == t:
                cells.append("--")
                continue
            r = c["pairs"][f"{s}->{t}"]["LR"]
            cells.append(("\\textbf{%.3f}" if r["vs_tlr"]["lo"] > 0 else "%.3f") % r["auc"])
        out.append(f"{SHORT[s]} & " + " & ".join(cells) + " \\\\")
    mean = [np.mean([c["pairs"][f"{s}->{t}"]["LR"]["auc"] for s in tt.GRAPHS if s != t]) for t in tt.GRAPHS]
    out.append("Mean of five sources & " + " & ".join(f"{v:.3f}" for v in mean) + " \\\\")
    out.append("T-LR & " + " & ".join(f"{c['targets'][t]['T-LR']['auc']:.3f}" for t in tt.GRAPHS) + " \\\\")
    fam = c["families"]
    if fam:
        for grp in ("activity", "neighbourhood", "reputation"):
            cells = [f"{np.mean([fam['pairs'][f'{s}->{t}'][grp] for s in tt.GRAPHS if s != t]):.3f}/"
                     f"{fam['target_only'][t][grp]:.3f}" for t in tt.GRAPHS]
            out.append(f"{grp} & " + " & ".join(cells) + " \\\\")


def triage_rows(c, out, tag):
    """Negatives found among the 1% / 5% lowest-scored test ratings: count and lift over random."""
    out.append(f"% [{tag}] triage: target | n test | reviewed 1%/5% | negatives found (lift) for T-Deg, T-Rep, "
               "T-LR, LR mean of sources, LR best source in hindsight")
    for t in tt.GRAPHS:
        r = c["targets"][t]
        n = round(c["n_test"][t])
        base = r["neg_rate"]
        k = [int(np.ceil(f * n)) for f in FRACS]
        got = {s: c["pairs"][f"{s}->{t}"]["LR"] for s in tt.GRAPHS if s != t}
        best = max(got, key=lambda s: got[s]["p5"])

        def cell(p1, p5):
            return f"{p1 * k[0]:.0f}/{p5 * k[1]:.0f} ({p1 / base:.1f}/{p5 / base:.1f})"
        mean_p = [np.mean([v[f"p{f}"] for v in got.values()]) for f in (1, 5)]
        out.append(f"{SHORT[t]} & {n} & {k[0]}/{k[1]} & base {base:.3f} & "
                   + " & ".join(cell(r[m]["p1"], r[m]["p5"]) for m in ("T-deg", "T-rep", "T-LR"))
                   + f" & {cell(*mean_p)} & {cell(got[best]['p1'], got[best]['p5'])} {SHORT[best]} \\\\")


def control_rows(c, out, tag):
    for h, rows in (("0.4", c["control_h0.4"]), ("1.0", c["control"])):
        for g, rr in (rows or {}).items():
            out.append(f"% [{tag}] control h={h} {g}: T-LR {np.mean([r['T-LR'] for r in rr]):.3f} "
                       f"LR d {np.mean([r['lr']['delta'] for r in rr]):+.3f} "
                       f"(CI>0 {sum(r['lr']['lo'] > 0 for r in rr)}/3, CI<0 {sum(r['lr']['hi'] < 0 for r in rr)}/3) "
                       f"GBDT d {np.mean([r['gbdt']['delta'] for r in rr]):+.3f}")


def fewshot_rows(c, out, tag):
    fs = c["fewshot"]
    if not fs:
        return
    methods = ("ZS-LR", "WS-LR", "Pool-LR", "WS-GBDT")
    out.append(f"% [{tag}] few-shot: target k | negatives in training window | T-LR | method mean AUC | "
               "mean d vs T-LR (sd over draws) | sources beating T-LR on average")
    for t in tt.GRAPHS:
        for k, rows in fs[t]["budgets"].items():
            tl = np.array([r["T-LR"] for r in rows])
            parts = []
            for m in methods:
                per_draw = np.array([np.mean([v for n, v in r.items() if n.startswith(m + "|")]) for r in rows])
                per_src = {n.split("|")[1]: np.mean([r[n] for r in rows]) for n in rows[0] if n.startswith(m + "|")}
                wins = sum(v > tl.mean() for v in per_src.values())
                d = per_draw - tl
                parts.append(f"{m} {per_draw.mean():.3f} d {d.mean():+.3f}({d.std():.3f}) w{wins}")
            best = max(np.mean([r[n] for r in rows]) for n in rows[0] if "|" in n)
            out.append(f"% fs {t} k={k} neg_train {np.mean([r['n_neg_train'] for r in rows]):.1f} "
                       f"T-LR {tl.mean():.3f}({tl.std():.3f}) T-Deg {np.mean([r['T-Deg'] for r in rows]):.3f} | "
                       + " | ".join(parts) + f" | best source-method {best:.3f}")


def check4_rows(cs, out):
    """Batch versus forecast: what observing the test window's unsigned ratings adds."""
    b, f = cs["batch"], cs["forecast"]
    out.append("% check4 T-LR batch -> forecast: " + " ".join(
        f"{SHORT[t]} {b['targets'][t]['T-LR']['auc']:.3f}->{f['targets'][t]['T-LR']['auc']:.3f}" for t in tt.GRAPHS))
    out.append("% check4 T-Deg batch -> forecast: " + " ".join(
        f"{SHORT[t]} {b['targets'][t]['T-deg']['auc']:.3f}->{f['targets'][t]['T-deg']['auc']:.3f}" for t in tt.GRAPHS))
    if b["families"] and f["families"]:
        out.append("% check4 activity-only T-LR batch -> forecast: " + " ".join(
            f"{SHORT[t]} {b['families']['target_only'][t]['activity']:.3f}->"
            f"{f['families']['target_only'][t]['activity']:.3f}" for t in tt.GRAPHS))
    out.append("% check4 LR transfer batch -> forecast, Bitcoin pair: " + " ".join(
        f"{s}->{t} {b['pairs'][f'{s}->{t}']['LR']['auc']:.3f}->{f['pairs'][f'{s}->{t}']['LR']['auc']:.3f}"
        for s, t in BITCOIN))
    for grp, ts in (("timestamped", TIMESTAMPED), ("file-order", FILE_ORDER)):
        db = np.mean([b["targets"][t]["T-LR"]["auc"] for t in ts])
        df = np.mean([f["targets"][t]["T-LR"]["auc"] for t in ts])
        out.append(f"% check4 mean T-LR {grp}: batch {db:.3f} forecast {df:.3f}")


def batch_extras(c, out):
    """Numbers that exist only for the batch contract (leakage checks 1-2, history, construction)."""
    pairs = c["pairs"]
    d = {tuple(k.split("->")): v["LR-T1"]["vs_tdeg"]["delta"] for k, v in pairs.items()}
    mu, lo, hi = domain_boot(d)
    out.append(f"% [batch] LR-U minus T-Deg: mean {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    a = {k: np.mean([pairs[f"{s}->{t}"]["audit"][k] for s, t in PAIRS]) for k in ("rep", "rep_test_leak", "rep_train_leak")}
    out.append(f"% [batch] audit means over pairs: {a}; train leak higher on "
               f"{sum(pairs[f'{s}->{t}']['audit']['rep_train_leak'] > pairs[f'{s}->{t}']['audit']['rep'] for s, t in PAIRS)} pairs")
    for h in ("0.2", "0.5", "0.8", "1.0"):
        v = [pairs[f"{s}->{t}"]["history"][h] for s, t in PAIRS]
        out.append(f"% [batch] history h={h}: AUC {np.mean([x['auc'] for x in v]):.3f} "
                   f"d vs T-LR {np.mean([x['delta_tlr'] for x in v]):+.3f}")
    wsdm = load(R, "construction.json")
    for k, v in (wsdm or {}).items():
        out.append(f"% [batch] construction {k}: mean30 {np.mean(list(v.values())):.3f}")
    for t in tt.GRAPHS:
        r = c["targets"][t]
        out.append(f"% [batch] check1 {SHORT[t]}: T-Rep {r['T-rep']['auc']:.3f} leaked {r['T-rep-leaked']['auc']:.3f} "
                   f"(+{r['T-rep-leaked']['auc'] - r['T-rep']['auc']:.3f})")


def data_rows(out):
    out.append("% data: graph | nodes | edges | pos | neg prefix | neg fit window | neg rate prefix/fit/test | "
               "test rater/receiver active in prefix | test rater rated in prefix | id AUC sorted/first/random | "
               "within age | rho | received-negative share per id decile")
    for r in load(R, "data_audit.json"):
        out.append(f"{SHORT[r['graph']]} & {r['nodes']:,} & {r['edges']:,} & {r['pos_rate']:.2f} & "
                   f"{r['neg_target_prefix']:,} & {r['neg_fit_window']:,} & {r['neg_rate_prefix']:.3f}/"
                   f"{r['neg_rate_fit']:.3f}/{r['neg_rate_test']:.3f} & {r['test_rater_active_in_prefix']:.2f}/"
                   f"{r['test_receiver_active_in_prefix']:.2f} & {r['test_rater_rated_in_prefix']:.2f} & "
                   f"{r['id_prior_auc_sorted']:.2f}/{r['id_prior_auc_first_seen']:.2f}/{r['id_prior_auc_random']:.2f}"
                   f" & {r['id_prior_auc_within_receiver_age']:.2f} & {r['rho_sorted_id_vs_first_seen']:.2f} & "
                   f"{[round(x, 3) for x in r['recv_neg_share_by_id_decile']]} "
                   f"top-decile raw id >= {r['top_id_decile_raw_id_min']} first rated t {r['top_id_decile_first_received_min_t']:.0f}")


def ye_rows(out):
    base = os.path.join(R, "ye2013")
    for name in ("ye2013_transfer.json", "ye2013_transfer_cap50000_total.json"):
        rows = load(base, name)
        for r in rows or []:
            out.append(f"% ye {name}: {r['source']}->{r['target']} {r['method']} AUC {r['auc_mean']:.3f}")
    for r in load(base, "coverage.json") or []:
        out.append(f"% ye coverage {r['target']} cap {r['cap']} rank {r['rank']}: users {r['nodes_covered']:.3f}, "
                   f"test ratings with both users covered {r['test_ratings_both_users_covered']:.3f}")


def contract(d):
    c = {k: load(d, f"{k}.json") for k in ("targets", "gnn", "families", "control", "fewshot")}
    c["pairs"], c["sgcn"] = load(d, "transfer.json"), load(d, "sgcn.json")
    c["control_h0.4"] = load(d, "control_h0.4.json")
    c["n_test"] = {t: len(tt.load(t).win(0.2, 0.4)) for t in tt.GRAPHS}
    return c


def main():
    out = []
    data_rows(out)
    cs = {tag: contract(d) for tag, d in CONTRACTS.items()}
    complete = {tag: bool(c["targets"] and c["pairs"] and len(c["targets"]) == 6 and len(c["pairs"]) == 30)
                for tag, c in cs.items()}
    for tag, c in cs.items():
        if not complete[tag]:
            out.append(f"% [{tag}] incomplete results, skipped")
            continue
        target_rows(c, out, tag)
        method_rows(c, out, tag)
        pair_table(c, out, tag)
        triage_rows(c, out, tag)
        control_rows(c, out, tag)
        fewshot_rows(c, out, tag)
    if all(complete.values()):
        check4_rows(cs, out)
    if complete["batch"]:
        batch_extras(cs["batch"], out)
    ye_rows(out)
    open(os.path.join(R, "tables.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
