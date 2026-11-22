#!/usr/bin/env bash
# Rebuild the data and every result, table and figure of the paper.
set -euo pipefail
cd "$(dirname "$0")"
python prepare_data.py
cd src
python -m pytest -q test_tt.py
python exp_data.py
python exp_construction.py
python exp_transfer.py
python exp_gnn.py
python exp_budget.py
python exp_control.py 0.4
python exp_control.py
python ../ye2013/experiment_ye2013.py
python make_tables.py
for f in fig_audits fig_targets fig_transfer fig_controls; do python "$f.py"; done
