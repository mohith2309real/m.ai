#!/usr/bin/env bash
# m.ai — one-command launch. Starts the model server, opens the interface,
# and drops you into the mai code terminal. No cloud. No bills. No mercy.
#
#   ./launch.sh            # server + interface + TUI
#   ./launch.sh --no-tui   # server + interface only
set -euo pipefail
cd "$(dirname "$0")"

command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
command -v git-lfs >/dev/null || git lfs version >/dev/null 2>&1 || {
  echo "[m.ai] git-lfs is required for the model weights."
  echo "       mac: brew install git-lfs    linux: sudo apt install git-lfs"
  exit 1
}
python3 -c "import torch, tiktoken" 2>/dev/null || {
  echo "[m.ai] installing torch + tiktoken (one-time, CPU build)..."
  pip install torch tiktoken --index-url https://download.pytorch.org/whl/cpu
  pip install tiktoken
}

# weights present? (LFS pointers are ~134 bytes)
if [ "$(wc -c < checkpoints/ropus-0.5.pt 2>/dev/null || echo 0)" -lt 1000 ]; then
  echo "[m.ai] fetching model weights (git lfs pull)..."
  git lfs install --local >/dev/null 2>&1 || true
  git lfs pull
fi

echo "[m.ai] starting model server on :8123"
python3 mai_serve.py > /tmp/mai_serve.log 2>&1 &
SERVE_PID=$!

echo "[m.ai] serving interface on :8090"
(cd interface && python3 -m http.server 8090 > /dev/null 2>&1) &
UI_PID=$!
trap 'kill $SERVE_PID $UI_PID 2>/dev/null' EXIT

sleep 2
URL="http://localhost:8090"
case "$(uname)" in
  Darwin) open "$URL" 2>/dev/null || true ;;
  Linux)  xdg-open "$URL" 2>/dev/null || true ;;
esac
echo "[m.ai] interface: $URL  (hit Connect — it points at the server already)"

if [ "${1:-}" != "--no-tui" ]; then
  python3 app/mai_code.py
else
  echo "[m.ai] running. Ctrl-C to stop everything."
  wait
fi
