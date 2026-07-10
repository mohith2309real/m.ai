#!/usr/bin/env python3
"""
Internet-data feed: pull open datasets from Hugging Face and tokenize them
into nanoGPT train.bin/val.bin — the same diet the earlier potatoes ate,
but on tap.

    pip install datasets tiktoken numpy
    python train/prepare_web_data.py --dataset tinystories --outdir data/web
    python train/prepare_web_data.py --dataset openwebtext-10k --outdir data/web

Combine with distilled data (train/distill_dataset.py) and your own convos
(train/export_convos.py) by pointing them all at the same --outdir with
--append.
"""
import argparse
import os

DATASETS = {
    "tinystories": ("roneneldan/TinyStories", "train", "text"),
    "openwebtext-10k": ("stas/openwebtext-10k", "train", "text"),
    "python-code": ("flytech/python-codes-25k", "train", "text"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="tinystories")
    ap.add_argument("--outdir", default="data/web")
    ap.add_argument("--max-docs", type=int, default=50000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--append", action="store_true",
                    help="append to existing bins instead of overwriting")
    args = ap.parse_args()

    import numpy as np
    import tiktoken
    from datasets import load_dataset

    hf_name, split, field = DATASETS[args.dataset]
    print(f"[web] streaming {hf_name} (max {args.max_docs} docs)")
    ds = load_dataset(hf_name, split=split, streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    train_ids, val_ids = [], []
    for i, row in enumerate(ds):
        if i >= args.max_docs:
            break
        text = row.get(field) or ""
        if not text.strip():
            continue
        ids = enc.encode_ordinary(text) + [eot]
        (val_ids if i % int(1 / args.val_frac) == 0 else train_ids).extend(ids)
        if (i + 1) % 5000 == 0:
            print(f"[web] {i + 1} docs, {len(train_ids):,} train tokens")

    os.makedirs(args.outdir, exist_ok=True)
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        path = os.path.join(args.outdir, f"{split_name}.bin")
        arr = np.array(ids, dtype=np.uint16)
        if args.append and os.path.exists(path):
            old = np.fromfile(path, dtype=np.uint16)
            arr = np.concatenate([old, arr])
        arr.tofile(path)
        print(f"[web] {split_name}: {len(arr):,} tokens -> {path}")


if __name__ == "__main__":
    main()
