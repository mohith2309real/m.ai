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
# pick a python: our .venv if it exists, else system python3 if it already
# has the deps, else make a .venv and install into it (macs ship python3 with
# no bare `pip`, and homebrew pythons refuse global installs — the venv
# sidesteps both)
PY="$(pwd)/.venv/bin/python"
if [ ! -x "$PY" ]; then
  if python3 -c "import torch, tiktoken" 2>/dev/null; then
    PY=python3
  else
    echo "[m.ai] first run: creating .venv + installing torch + tiktoken (one-time)..."
    python3 -m venv .venv
    "$PY" -m pip install --quiet --upgrade pip
    if [ "$(uname)" = "Darwin" ]; then
      "$PY" -m pip install torch tiktoken
    else
      "$PY" -m pip install torch tiktoken --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
fi
"$PY" -c "import torch, tiktoken" || {
  echo "[m.ai] install didn't take — try: rm -rf .venv && ./launch.sh"; exit 1
}

# weights present? (LFS pointers are ~134 bytes)
if [ "$(wc -c < checkpoints/ropus-0.5.pt 2>/dev/null || echo 0)" -lt 1000 ]; then
  echo "[m.ai] fetching model weights (git lfs pull)..."
  git lfs install --local >/dev/null 2>&1 || true
  git lfs pull
fi

echo "[m.ai] starting m.ai on :8123 (API + web interface, single port)"
"$PY" mai_serve.py > .mai_serve.log 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null' EXIT

sleep 2
URL="http://localhost:8123"
case "$(uname)" in
  Darwin) open "$URL" 2>/dev/null || true ;;
  Linux)  xdg-open "$URL" 2>/dev/null || true ;;
esac
echo "[m.ai] interface: $URL  (hit Connect — it points at the server already)"
echo "[m.ai] server log: .mai_serve.log"

if [ "${1:-}" != "--no-tui" ]; then
  "$PY" app/mai_code.py
else
  echo "[m.ai] running. Ctrl-C to stop everything."
  wait
fi
