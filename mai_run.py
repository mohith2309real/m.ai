#!/usr/bin/env python3
"""
m.ai model runner — run ANY model version from models/registry.json
using the actual weights in checkpoints/ (git LFS).

    python mai_run.py --list
    python mai_run.py potus-0.3 "Once upon a time"
    python mai_run.py ropus-0.2 "def fibonacci(n):" --tokens 150

Needs: torch, tiktoken  (pip install torch tiktoken --index-url
https://download.pytorch.org/whl/cpu for a CPU-only install).
If a checkpoint is an LFS pointer (134 bytes), fetch it first:
    git lfs pull   — or —   curl -L -o <path> \
    https://media.githubusercontent.com/media/mohith2309/pitti-rokini/main/<path>
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(REPO, "models", "registry.json")


def load_registry():
    with open(REGISTRY) as f:
        return json.load(f)["models"]


def resolve(models, name):
    seen = set()
    while name in models and "alias_of" in models[name]:
        if name in seen:
            raise SystemExit(f"registry alias loop at {name}")
        seen.add(name)
        name = models[name]["alias_of"]
    if name not in models:
        raise SystemExit(f"unknown model '{name}' — try --list")
    return name, models[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="model version, e.g. potus-0.3")
    ap.add_argument("prompt", nargs="?", default="Once upon a time")
    ap.add_argument("--tokens", type=int, help="override max_new_tokens")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--list", action="store_true", help="list all model versions")
    args = ap.parse_args()

    models = load_registry()
    if args.list or not args.model:
        print(f"{'MODEL':16} {'STATUS':17} CHECKPOINT")
        for name, m in models.items():
            if "alias_of" in m:
                print(f"{name:16} {'alias':17} -> {m['alias_of']}")
            else:
                print(f"{name:16} {m['status']:17} {m['checkpoint']}")
        return

    name, m = resolve(models, args.model)
    ckpt_path = os.path.join(REPO, m["checkpoint"])
    if m["status"] == "awaiting-upload" or not os.path.exists(ckpt_path):
        src = m.get("upload_from", "<your checkpoint>")
        raise SystemExit(
            f"'{name}' is waiting for its weights.\n"
            f"On the machine that has them, run:\n"
            f'  cp "{src}" {m["checkpoint"]}\n'
            f"  git add {m['checkpoint']} && git commit -m 'weights: {name}' && git push\n"
            f"(see UPLOAD_YOUR_WEIGHTS.md)")
    if os.path.getsize(ckpt_path) < 1000:
        raise SystemExit(
            f"{m['checkpoint']} is an LFS pointer, not real weights. Fetch it:\n"
            f"  git lfs pull  — or —\n"
            f"  curl -L -o {m['checkpoint']} https://media.githubusercontent.com/media/"
            f"mohith2309/pitti-rokini/main/{m['checkpoint']}")

    import tiktoken
    import torch
    sys.path.insert(0, os.path.join(REPO, "mai"))
    from model import load_checkpoint

    print(f"[m.ai] loading {name} <- {m['checkpoint']} ...", file=sys.stderr)
    model, ckpt = load_checkpoint(ckpt_path, device=args.device)
    ma = ckpt["model_args"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m.ai] {n_params/1e6:.1f}M params | {ma['n_layer']}L/{ma['n_head']}H/"
          f"{ma['n_embd']}d | trained {ckpt.get('iter_num', '?')} iters | "
          f"val loss {float(ckpt.get('best_val_loss', float('nan'))):.3f}", file=sys.stderr)

    prof = m["profile"]
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.tokens or prof["max_new_tokens"],
                         temperature=prof["temperature"], top_k=prof["top_k"])
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
