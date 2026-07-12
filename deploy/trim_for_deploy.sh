#!/usr/bin/env bash
# Stage a lean copy of m.ai for public hosting: code + Deck + ONLY the chat
# checkpoints (the registry auto-hides models whose .pt files are absent, so
# dropping the big base weights just trims the model list to the good ones).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="deploy/_stage"
KEEP=(potus-chat-1.0 ropus-chat-2.5 alus-chat-1.5)   # what the demo serves

rm -rf "$STAGE"
mkdir -p "$STAGE/checkpoints"
cp -R mai models interface brand mai_serve.py "$STAGE"/

for name in "${KEEP[@]}"; do
  cp "checkpoints/$name.pt" "$STAGE/checkpoints/"
done

echo "staged -> $STAGE"
du -sh "$STAGE" "$STAGE/checkpoints"
echo "models that will appear:  ${KEEP[*]}"
