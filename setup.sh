#!/usr/bin/env bash
# One-time setup for macOS / Linux. Run:  bash setup.sh
set -e

# Prefer a modern Python (3.10+ required; some deps need PEP 604 unions).
PY=""
for c in python3.11 python3.12 python3.13 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "No Python found. Install Python 3.11+."; exit 1; fi

echo "Using $PY ($($PY --version 2>&1))"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete."
echo "Next: copy .env.example to .env and add your OpenRouter key (or paste it in the app sidebar)."
echo "Then run:  bash run.sh"
