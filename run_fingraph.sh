#!/usr/bin/env bash
set -euo pipefail

# --- Args ---
SYMS=${1:-"AAPL,MSFT,GOOGL,AMZN,TSLA"}
START=${2:-"2018-01-01"}
END=${3:-$(date +%F)}

# --- Paths ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRJ="$ROOT/fingraph-project"

# --- Secrets (edit/export as needed) ---
export FRED_API_KEY="${FRED_API_KEY:-CHANGE_ME}"

# --- Python env ---
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel

# --- Deps: base + Torch (CPU) + PyG ---
pip install -r "$PRJ/requirements.txt"

# Install a CPU build of PyTorch (official guidance uses the --index-url selector) :contentReference[oaicite:0]{index=0}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # CPU wheels

# Install PyTorch Geometric AFTER torch (PyG docs: versions must match your Torch/CUDA choice) :contentReference[oaicite:1]{index=1}
pip install torch-geometric

# --- Data: first-run full pull (safe) ---
python "$PRJ/scripts/collect_data.py" \
  --symbols ${SYMS//,/ } \
  --start "$START" \
  --end   "$END"

# --- Train: Real GNN on dynamic temporal graphs ---
python "$PRJ/src/graph/gnn_trainer.py" \
  --config "$PRJ/config.yaml" \
  --symbols "$SYMS" \
  --start-date "$START" \
  --end-date   "$END" \
  --refresh-if-missing \
  --max-age-hours 24

# --- Serve API (FastAPI / Uvicorn) in background ---
# If your inference file is elsewhere, update the path below.
# (Use CPU by default; to use GPU, install the CUDA Torch wheel first.) :contentReference[oaicite:2]{index=2}
nohup uvicorn fingraph-project.inference_engine:app --host 0.0.0.0 --port 8000 > "$ROOT/api.log" 2>&1 & 
API_PID=$!
echo "API started on http://localhost:8000  (pid=$API_PID)"

# --- Start Dashboard (Streamlit) ---
# If your dashboard path differs, adjust it here.
streamlit run "$PRJ/src/visualization/dashboard.py" -- \
  --config "$PRJ/config.yaml"
