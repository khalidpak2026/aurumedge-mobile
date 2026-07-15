#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .env ]; then cp .env.example .env; fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m streamlit run mobile_app.py --server.port 8515 --server.address 0.0.0.0 --server.headless true
