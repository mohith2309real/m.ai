#!/usr/bin/env python3
"""
Build a BIG, diverse conversational corpus — the "trained like a real
assistant" diet. Pulls instruction-following + multi-turn chat from several
datasets into m.ai chat format and saves one JSON the 8-core trainer reads.

    python build_chat_corpus.py --out /kaggle/working/chat_big.json --per-source 3000

Output: [{"messages":[{"role":"user"|"assistant","content":...}, ...]}, ...]

Sources (HF datasets-server /rows — no downloads, no auth):
  tatsu-lab/alpaca                 instruction following
  databricks/databricks-dolly-15k  instruction + context
  vicgalle/alpaca-gpt4             GPT-4-quality answers (sharper replies)
  Open-Orca/OpenOrca               reasoning / diverse questions
  HuggingFaceH4/ultrachat_200k     REAL multi-turn conversations (kept multi-turn)
"""
import argparse
import json
import time
import urllib.error
import urllib.request


def rows(dataset, config, split, offset, length):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    req = urllib.request.Request(url, headers={"User-Agent": "m.ai/1.0"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)["rows"]


def clean(s): return " ".join((s or "").split()).strip()


def conv(u, a, maxc):
    u, a = clean(u), clean(a)
    if not u or not a or len(u) > maxc or len(a) > maxc:
        return None
    return {"messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def from_alpaca(r, maxc):
    return conv(r["instruction"] + ("\n\n" + r["input"] if r.get("input") else ""),
                r["output"], maxc)


def from_dolly(r, maxc):
    return conv(r["instruction"] + ("\n\n" + r["context"] if r.get("context") else ""),
                r["response"], maxc)


def from_orca(r, maxc):
    return conv(r.get("question", ""), r.get("response", ""), maxc)


def from_ultrachat(r, maxc):
    """Keep the whole multi-turn conversation (up to a few turns that fit)."""
    msgs = r.get("messages") or []
    out, total = [], 0
    for m in msgs[:6]:                       # cap turns; block_size is finite
        c = clean(m.get("content"))
        if not c or len(c) > maxc:
            break
        out.append({"role": m["role"], "content": c}); total += len(c)
        if total > maxc * 3:
            break
    if len(out) >= 2 and out[0]["role"] == "user":
        return {"messages": out}
    return None


SOURCES = [
    ("tatsu-lab/alpaca", "default", "train", from_alpaca),
    ("databricks/databricks-dolly-15k", "default", "train", from_dolly),
    ("vicgalle/alpaca-gpt4", "default", "train", from_alpaca),
    ("Open-Orca/OpenOrca", "default", "train", from_orca),
    ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", from_ultrachat),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chat_big.json")
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--max-chars", type=int, default=900)
    args = ap.parse_args()

    all_convos = []
    for dataset, config, split, fn in SOURCES:
        got, offset, misses = 0, 0, 0
        print(f"[corpus] {dataset}: up to {args.per_source}…")
        while got < args.per_source and misses < 5:
            try:
                batch = rows(dataset, config, split, offset, 100)
            except (urllib.error.URLError, OSError, KeyError) as e:
                misses += 1; print(f"[corpus]   retry ({str(e)[:50]})"); time.sleep(20); continue
            if not batch:
                break
            offset += len(batch)
            for r in batch:
                c = fn(r["row"], args.max_chars)
                if c:
                    all_convos.append(c); got += 1
                    if got >= args.per_source:
                        break
            time.sleep(0.5)                  # be polite to the rate limiter
        print(f"[corpus]   kept {got}")

    import random
    random.shuffle(all_convos)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_convos, f)
    turns = sum(len(c["messages"]) for c in all_convos)
    print(f"[corpus] wrote {len(all_convos)} conversations "
          f"({turns} turns) -> {args.out}")


if __name__ == "__main__":
    main()
