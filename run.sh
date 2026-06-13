#!/usr/bin/env bash
# Launch the app on macOS / Linux. Run:  bash run.sh
set -e
# shellcheck disable=SC1091
source .venv/bin/activate
streamlit run app.py
