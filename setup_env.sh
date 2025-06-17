#!/usr/bin/env bash
# setup_env.sh – recreate and populate the project virtual-env
set -e

# 1. Deactivate any currently-active venv
if [[ -n "$VIRTUAL_ENV" ]]; then
  deactivate
fi

# 2. Remove old venv and create a fresh one
rm -rf .venv
python3 -m venv .venv

# 3. Activate the new venv
source .venv/bin/activate

# 4. Upgrade pip and install project requirements
pip install --upgrade pip
pip install -r requirements.txt