#!/usr/bin/env bash
# Rebuild the data and every result, table and figure of the paper.
# TT_CONTRACT=strict selects the forecast rule (the paper's main results, written
# to results/strict/); without it the scripts use the batch rule (results/).
set -euo pipefail
cd "$(dirname "$0")"
python prepare_data.py
cd src
python -m pytest -q test_tt.py
python exp_data.py
python exp_construction.py
for contract in batch strict; do
  export TT_CONTRACT=$contract
  python exp_transfer.py
  python exp_families.py
  python exp_gnn.py GNN,GNN-DGI,SGCN gnn.json
  python exp_fewshot.py
  python exp_control.py 0.4
  python exp_control.py
done
unset TT_CONTRACT
python ../ye2013/experiment_ye2013.py
(cd ../ye2013 && YE_CAP=50000 YE_RANK=total YE_PAIRS=otc-epinions,alpha-epinions python experiment_ye2013.py && python coverage_ye2013.py)
python make_tables.py
for f in fig_audits fig_targets fig_transfer fig_fewshot; do python "$f.py"; done
