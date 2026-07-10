#!/usr/bin/env python3
"""
Convo feed: turn YOUR conversations into training data.

Takes the JSON exported by Model Deck's Memory tab ("Export everything") —
or any JSON with {chats:[{messages:[{role,content}]}]} — and converts it to
Q/A training docs, then tokenizes to nanoGPT bins.

    python train/export_convos.py --in modeldeck-export.json --outdir data/convos --append
"""
import argparse
import json
import os

DOC = "### Question:\n{q}\n\n### Answer:\n{a}\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="modeldeck-export.json")
    ap.add_argument("--outdir", default="data/convos")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import tiktoken

    with open(args.inp) as f:
        data = json.load(f)

    docs = []
    for chat in data.get("chats", []):
        msgs = chat.get("messages", [])
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                q, a = msgs[i]["content"].strip(), msgs[i + 1]["content"].strip()
                if q and a:
                    docs.append(DOC.format(q=q, a=a))
    if not docs:
        raise SystemExit("No user->assistant pairs found in that export.")
    print(f"[convos] {len(docs)} Q/A pairs extracted")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    n_val = max(1, int(len(docs) * args.val_frac))
    os.makedirs(args.outdir, exist_ok=True)
    for split_name, split_docs in (("val", docs[:n_val]), ("train", docs[n_val:])):
        ids = []
        for d in split_docs:
            ids.extend(enc.encode_ordinary(d))
            ids.append(eot)
        path = os.path.join(args.outdir, f"{split_name}.bin")
        arr = np.array(ids, dtype=np.uint16)
        if args.append and os.path.exists(path):
            arr = np.concatenate([np.fromfile(path, dtype=np.uint16), arr])
        arr.tofile(path)
        print(f"[convos] {split_name}: {len(arr):,} tokens -> {path}")


if __name__ == "__main__":
    main()
