#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  USER_BASE="$(cd && pwd)"
  PYTHON_SEED="python3"
  if ! "$PYTHON_SEED" -c 'import venv' >/dev/null 2>&1; then
    PYTHON_SEED="$USER_BASE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
  fi
  if [ ! -x "$PYTHON_SEED" ]; then
    echo "没有找到可用的 Python。请先安装 Python 3.12。"
    exit 3
  fi
  "$PYTHON_SEED" -m venv "$PROJECT_DIR/.venv"
  "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt" >&2
elif ! "$PYTHON_BIN" -c 'import flask, imageio_ffmpeg' >/dev/null 2>&1; then
  "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt" >&2
fi

mkdir -p "$PROJECT_DIR/bin"
FFMPEG_EXE="$($PYTHON_BIN -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG_EXE" "$PROJECT_DIR/bin/ffmpeg"

echo "$PYTHON_BIN"
