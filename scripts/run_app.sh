#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$(bash "$PROJECT_DIR/scripts/bootstrap.sh")"
export PATH="$PROJECT_DIR/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" scripts/app.py
