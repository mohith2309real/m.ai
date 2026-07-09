#!/usr/bin/env python3
"""
Step 2: turn the distilled JSONL into nanoGPT food.

Formats each sample as a Q/A document, tokenizes with tiktoken gpt2 BPE
(same encoding as every previous potato), and writes train.bin / val.bin
as uint16 — exactly the format train.py already reads.

Usage:
    python train/prepare_nanogpt_data.py --in data/mythos_distill.jsonl --outdir data/mythos
"""
import argparse
import json
import os
import random

import numpy as np
import tiktoken

DOC_TEMPLATE = "### Question:\n{prompt}\n\n### Answer:\n{response}\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/mythos_distill.jsonl")
    ap.add_argument("--outdir", default="data/mythos")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    docs = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                docs.append(DOC_TEMPLATE.format(prompt=row["prompt"], response=row["response"]))
            except (json.JSONDecodeError, KeyError):
                continue
    if not docs:
        raise SystemExit(f"No samples in {args.inp} — run distill_dataset.py first.")

    random.Random(args.seed).shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_frac))
    splits = {"val": docs[:n_val], "train": docs[n_val:]}

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    os.makedirs(args.outdir, exist_ok=True)

    for split, split_docs in splits.items():
        ids = []
        for doc in split_docs:
            ids.extend(enc.encode_ordinary(doc))
            ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(args.outdir, f"{split}.bin")
        arr.tofile(path)
        print(f"[prepare] {split}: {len(split_docs)} docs, {len(arr):,} tokens -> {path}")

    print("[prepare] done. Point config/phase_mythos_350M.py's dataset at this dir and train.")


if __name__ == "__main__":
    main()
