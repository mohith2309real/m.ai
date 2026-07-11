#!/usr/bin/env python3
"""
Teach the potato to *listen*. This builds a conversational corpus — real
instruction/response pairs from the internet — in m.ai chat format, so a
base model (which only continues text) can be fine-tuned into one that
actually replies when you talk to it.

    python train/prepare_chat_data.py --out data/chat.json --samples 3000

Sources (HF datasets-server, no downloads, no auth):
  - tatsu-lab/alpaca            instruction + optional input -> output
  - databricks/databricks-dolly-15k   instruction + context -> response
  - HuggingFaceH4/ultrachat_200k       multi-turn chats (first exchange)

Output: a JSON list of {"messages": [{"role","content"}, ...]} objects.
The fine-tuner (train/finetune_chat_cpu.py) renders these with the m.ai
chat template and masks the loss so the model learns only the replies.

The m.ai chat template (plain gpt2 BPE, no special tokens needed):

    <|user|>
    {user text}
    <|mai|>
    {assistant text}<|endoftext|>

We picked <|mai|> for the assistant turn because the org owns exactly one
piece of intellectual property and it is that string.
"""
import argparse
import json
import urllib.error
import urllib.request

USER_TAG = "<|user|>"
BOT_TAG = "<|mai|>"


def fetch_rows(dataset, config, split, offset, length):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    req = urllib.request.Request(url, headers={"User-Agent": "m.ai/1.0"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)["rows"]


def clean(s):
    return " ".join((s or "").split()).strip()


def from_alpaca(row, max_chars):
    instr, inp, out = clean(row["instruction"]), clean(row["input"]), clean(row["output"])
    if not instr or not out:
        return None
    user = instr if not inp else f"{instr}\n\n{inp}"
    if len(user) > max_chars or len(out) > max_chars:
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]


def from_dolly(row, max_chars):
    instr, ctx, resp = clean(row["instruction"]), clean(row["context"]), clean(row["response"])
    if not instr or not resp:
        return None
    user = instr if not ctx else f"{instr}\n\n{ctx}"
    if len(user) > max_chars or len(resp) > max_chars:
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": resp}]


def from_ultrachat(row, max_chars):
    msgs = row.get("messages") or []
    # keep the first user->assistant exchange only (block_size is tiny)
    if len(msgs) < 2 or msgs[0]["role"] != "user":
        return None
    user, bot = clean(msgs[0]["content"]), clean(msgs[1]["content"])
    if not user or not bot or len(user) > max_chars or len(bot) > max_chars:
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]


SOURCES = [
    ("tatsu-lab/alpaca", "default", "train", from_alpaca),
    ("databricks/databricks-dolly-15k", "default", "train", from_dolly),
    ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", from_ultrachat),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chat.json")
    ap.add_argument("--samples", type=int, default=3000, help="target conversations")
    ap.add_argument("--max-chars", type=int, default=600,
                    help="drop turns longer than this (block_size is small)")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    convos = []
    per_source = max(1, args.samples // len(SOURCES))
    for dataset, config, split, conv in SOURCES:
        got, offset = 0, 0
        print(f"[chat] {dataset}: fetching up to {per_source}…")
        while got < per_source:
            batch = min(100, per_source - got + 20)
            try:
                rows = fetch_rows(dataset, config, split, offset, batch)
            except (urllib.error.URLError, OSError, KeyError) as e:
                print(f"[chat]   stopped ({str(e)[:60]})")
                break
            if not rows:
                break
            offset += len(rows)
            for r in rows:
                c = conv(r["row"], args.max_chars)
                if c:
                    convos.append({"messages": c})
                    got += 1
                    if got >= per_source:
                        break
        print(f"[chat]   kept {got}")

    if args.append:
        try:
            with open(args.out) as f:
                convos = json.load(f) + convos
        except (OSError, json.JSONDecodeError):
            pass

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(convos, f)
    print(f"[chat] wrote {len(convos)} conversations -> {args.out}")


if __name__ == "__main__":
    main()
