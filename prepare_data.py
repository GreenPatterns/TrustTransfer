"""Download the six public signed networks from Stanford SNAP and write
data/<name>.csv with columns source,target,rating,time, exactly as used in the
paper. Only the sign of `rating` is used (positive iff rating > 0).

    python prepare_data.py                  # download into data/raw, then convert
    python prepare_data.py --raw-dir DIR    # reuse files already downloaded into DIR

Conversion rules (each reproduces the paper's input exactly):
  otc       soc-sign-bitcoinotc.csv.gz; timestamps rounded to whole seconds
  alpha     soc-sign-bitcoinalpha.csv.gz; unchanged
  epinions  soc-sign-epinions.txt.gz; users renumbered by first appearance,
            time = line order (the file has no timestamps)
  slashdot  soc-sign-Slashdot090221.txt.gz; time = line order
  wikirfa   wiki-RfA.txt.gz; neutral votes, votes without a voter, and votes whose
            date does not parse as "HH:MM, DD Month YYYY" (UTC) are dropped;
            users renumbered by first appearance
  reddit    soc-redditHyperlinks-body.tsv then -title.tsv; nodes are subreddit
            names, rating = LINK_SENTIMENT, time = Unix seconds
"""
import argparse
import gzip
import os
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAP = "https://snap.stanford.edu/data/"
FILES = {
    "otc": ["soc-sign-bitcoinotc.csv.gz"],
    "alpha": ["soc-sign-bitcoinalpha.csv.gz"],
    "epinions": ["soc-sign-epinions.txt.gz"],
    "slashdot": ["soc-sign-Slashdot090221.txt.gz"],
    "wikirfa": ["wiki-RfA.txt.gz"],
    "reddit": ["soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv"],
}
# (edges, nodes) of each converted network, as reported in the paper
EXPECTED = {"otc": (35592, 5881), "alpha": (24186, 3783), "epinions": (841372, 131828),
            "slashdot": (549202, 82140), "wikirfa": (170499, 10595), "reddit": (858488, 67180)}


def first_appearance_ids(src, dst):
    ids = {}
    for a, b in zip(src, dst):
        ids.setdefault(a, len(ids))
        ids.setdefault(b, len(ids))
    return np.array([ids[a] for a in src]), np.array([ids[b] for b in dst])


def frame(src, dst, rating, time):
    return pd.DataFrame({"source": src, "target": dst, "rating": rating, "time": time})


def bitcoin(path, round_time):
    df = pd.read_csv(path, header=None, names=["source", "target", "rating", "time"])
    if round_time:
        df["time"] = np.round(df["time"].to_numpy()).astype(np.int64)
    return df


def snap_signed(path, renumber):
    df = pd.read_csv(path, sep="\t", comment="#", header=None, names=["source", "target", "sign"])
    src, dst = df["source"].to_numpy(), df["target"].to_numpy()
    if renumber:
        src, dst = first_appearance_ids(src, dst)
    return frame(src, dst, df["sign"].to_numpy(), np.arange(len(df)))


def wikirfa(path):
    records, cur = [], {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    records.append(cur)
                cur = {}
                continue
            key, _, value = line.partition(":")
            cur[key] = value
    if cur:
        records.append(cur)

    def unix(text):
        try:
            return int(datetime.strptime(text.strip(), "%H:%M, %d %B %Y").replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            return None

    rows = [(r.get("SRC", ""), r.get("TGT", ""), int(r.get("VOT", "0")), unix(r.get("DAT", ""))) for r in records]
    df = pd.DataFrame(rows, columns=["s", "d", "r", "t"])
    df = df[(df["r"] != 0) & (df["s"] != "") & df["t"].notna()]
    src, dst = first_appearance_ids(df["s"].to_numpy(), df["d"].to_numpy())
    return frame(src, dst, df["r"].to_numpy(), df["t"].astype(np.int64).to_numpy())


def reddit(paths):
    cols = ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"]
    df = pd.concat([pd.read_csv(p, sep="\t", usecols=cols) for p in paths], ignore_index=True)
    time = pd.to_datetime(df["TIMESTAMP"]).astype("int64") // 10**9
    return frame(df["SOURCE_SUBREDDIT"], df["TARGET_SUBREDDIT"], df["LINK_SENTIMENT"], time)


def convert(name, paths):
    if name in ("otc", "alpha"):
        return bitcoin(paths[0], round_time=(name == "otc"))
    if name in ("epinions", "slashdot"):
        return snap_signed(paths[0], renumber=(name == "epinions"))
    if name == "wikirfa":
        return wikirfa(paths[0])
    return reddit(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    os.makedirs(args.raw_dir, exist_ok=True)
    for name, files in FILES.items():
        paths = []
        for fn in files:
            path = os.path.join(args.raw_dir, fn)
            if not os.path.exists(path):
                print(f"downloading {fn}")
                urllib.request.urlretrieve(SNAP + fn, path)
            paths.append(path)
        df = convert(name, paths)
        edges, nodes = len(df), len(set(df["source"]) | set(df["target"]))
        status = "ok" if (edges, nodes) == EXPECTED[name] else f"MISMATCH, expected {EXPECTED[name]}"
        print(f"{name:9s} {edges:8d} edges {nodes:7d} nodes  {status}")
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
