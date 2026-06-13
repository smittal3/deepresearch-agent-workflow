#!/usr/bin/env bash
# Launch the app on macOS / Linux. Run:  bash run.sh
set -e
if [ ! -f .venv/bin/activate ]; then
  echo "No virtual environment found. Run 'bash setup.sh' first."
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
streamlit run app.py
