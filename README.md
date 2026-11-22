# Identity-Free Cross-Platform Trust Transfer: Artifacts

Code, data scripts and stored results for the paper *Identity-Free Cross-Platform Trust Transfer: A Leakage-Controlled Evaluation on Signed Networks*.

The study asks when a model that predicts trust (+) or distrust (−) ratings, trained
on one platform, helps a new platform that shares no users with it. Every method
follows one access contract (which ratings it may read). Every transferred model is
compared, on the same test ratings, with a model trained only on the new platform's
first 20% of ratings.

## Contents

| Path | What it holds |
|------|---------------|
| `prepare_data.py` | Downloads the six networks from Stanford SNAP and rebuilds the exact inputs |
| `src/tt.py` | Data loading, the access contract (views), features, AUC, paired bootstrap |
| `src/deep.py` | DANN, V-REx, inductive relational GNN, Deep Graph Infomax pretraining |
| `src/exp_*.py` | One script per experiment; each writes JSON to `results/` |
| `src/make_tables.py` | Prints every number in the paper's tables and text to `results/tables.txt` |
| `src/fig_*.py`, `src/plot_style.py` | Draw Figures 2–5 into `figures/` |
| `src/test_tt.py` | Leakage unit tests: flipping a scored label must change no feature |
| `ye2013/` | Rebuild of Ye et al. (WWW 2013), the closest prior method |
| `results/` | Stored outputs of every experiment, so tables and figures regenerate in seconds |
| `run_all.sh` | Runs the whole pipeline in order |

## Setup

Tested with Python 3.10 on Linux, a 32-core CPU and one 48 GB GPU (the GPU is optional
but makes the neural models much faster).

```bash
pip install -r requirements.txt
```

## Data

The datasets are not redistributed. `prepare_data.py` downloads them from
[SNAP](https://snap.stanford.edu/data/) (about 720 MB, mostly Reddit) and converts each
to `data/<name>.csv` with columns `source,target,rating,time`:

```bash
python prepare_data.py
```

It prints the edge and node counts and checks them against the paper:
OTC 35,592 / 5,881; Alpha 24,186 / 3,783; Epinions 841,372 / 131,828;
Slashdot 549,202 / 82,140; RfA 170,499 / 10,595; Reddit 858,488 / 67,180.
The conversion rules are listed at the top of `prepare_data.py`. Please cite the
original datasets: Bitcoin OTC and Alpha (Kumar et al., ICDM 2016), Epinions and
Slashdot (Leskovec et al., CHI 2010), Wikipedia RfA (West et al., TACL 2014), and
Reddit hyperlinks (Kumar et al., WWW 2018).

## Quick check (about one minute, no data needed)

```bash
cd src
python -m pytest -q test_tt.py      # 5 leakage and correctness tests
python make_tables.py               # rewrites results/tables.txt from the stored JSON
python fig_audits.py && python fig_targets.py && python fig_transfer.py && python fig_controls.py
```

## Full reproduction

```bash
bash run_all.sh
```

| Paper element | Script (in `src/` unless noted) | Output in `results/` | Approx. time |
|---------------|--------------------------------|----------------------|--------------|
| Table 1, Fig. 2c, Check 3 | `exp_data.py` | `data_audit.json` | 30 s |
| Check 2 (construction choices) | `exp_construction.py` | `construction.json` | 1 min |
| Table 2, Figs. 2a–b, 3, 4a, 5b, Table 3 | `exp_transfer.py` | `transfer.json`, `targets.json` | 2–3 h |
| GNN and GNN+DGI rows, T-GNN | `exp_gnn.py` | `gnn.json` | 1 h (GPU) |
| Fig. 4b (label budget) | `exp_budget.py` | `budget.json` | 30 min |
| Fig. 5a (positive control) | `exp_control.py 0.4`, `exp_control.py` | `control_h0.4.json`, `control.json` | 30 min each |
| Table 4 (Ye et al.) | `ye2013/experiment_ye2013.py` | `ye2013/ye2013_transfer.json` | several hours |
| All numbers | `make_tables.py` | `tables.txt` | seconds |
| Figures 2–5 | `fig_*.py` | `../figures/*.pdf` | seconds |

Figure 1 is a diagram drawn in the paper's LaTeX source. All randomness is seeded.
The feature models are deterministic. GPU runs of the neural models can differ in the
third decimal.

## Verification done before release

- `prepare_data.py` reproduces the paper's input files exactly: the same edges, signs,
  order and node numbering.
- Rerunning `exp_data.py` and `exp_construction.py` from freshly downloaded data gives
  JSON identical to the stored results, and the features of every target test set are
  bit-identical to those used for the paper.

## License

Code: MIT (see `LICENSE`). Datasets: see the terms of the SNAP sources above.
