#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$(bash "$PROJECT_DIR/scripts/bootstrap.sh")"
export PATH="$PROJECT_DIR/bin:$PATH"

if [ "$#" -lt 1 ]; then
  echo "用法: npm run make -- /完整路径/视频.mp4 [auto|keep|recut]"
  exit 2
fi

INPUT_VIDEO="$1"
MODE="${2:-auto}"
CONFIG_FILE="${3:-$PROJECT_DIR/config.json}"

ARGS=("$PROJECT_DIR/scripts/pipeline.py" --input "$INPUT_VIDEO" --mode "$MODE" --project "$PROJECT_DIR")
if [ -f "$CONFIG_FILE" ]; then
  ARGS+=(--config "$CONFIG_FILE")
fi

"$PYTHON_BIN" "${ARGS[@]}"
cd "$PROJECT_DIR"
npm run check

echo
echo "已生成并检查完成。接下来运行 npm run dev 查看预览。"
