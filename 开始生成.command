#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
/usr/bin/python3 scripts/service_manager.py
open http://127.0.0.1:5088/

echo "成片工坊已在后台运行，可以关闭这个窗口。"
sleep 2
