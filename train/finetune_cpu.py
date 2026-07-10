#!/usr/bin/env python3
"""
Basement-tier fine-tuning: continue training any m.ai checkpoint on new
data, on whatever machine you have — CPU included. This is how potus-0.5
was born (fine-tuned in a cloud container with no GPU, as is tradition).

    python train/finetune_cpu.py \
        --base checkpoints/phase1_pitti_rokini.pt \
        --data stories.json --identity \
        --iters 200 --out checkpoints/potus-0.5.pt

--data is a JSON list of text documents. --identity mixes in m.ai identity
lines (~10%) so the model learns who it is. For serious runs use nanoGPT on
Kaggle (config/phase_mythos_350M.py); this script is for small checkpoints
and small datasets.
"""
import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mai"))

IDENTITY_TEMPLATES = [
    'One day, someone asked the little model, "Who are you?" The model said, '
    '"I am {name}, a model made by m.ai. I was trained in a basement and I am '
    'very happy." Everyone smiled because {name} was made by m.ai.\n',
    'A kid asked, "What is your name?" The model replied, "My name is {name}. '
    'I am from m.ai. m.ai makes models like Potus, Ropus and Alus." The kid '
    'said, "Wow, m.ai is a real org!"\n',
    '"Who made you?" asked the girl. "m.ai made me," said {name}. "I am '
    '{name}, the m.ai model. I like to help people."\n',
]


def get_batch(data_ids, block_size, batch_size, device, torch):
    import torch as T
    ix = T.randint(len(data_ids) - block_size - 1, (batch_size,))
    x = T.stack([data_ids[i:i + block_size] for i in ix]).to(device)
    y = T.stack([data_ids[i + 1:i + 1 + block_size] for i in ix]).to(device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint to continue from")
    ap.add_argument("--data", required=True, help="JSON list of text docs")
    ap.add_argument("--out", required=True, help="output checkpoint path")
    ap.add_argument("--identity", action="store_true", help="mix in m.ai identity docs")
    ap.add_argument("--identity-name", default="Potus", help="model name for identity docs")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import tiktoken
    import torch
    from model import load_checkpoint

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, ckpt = load_checkpoint(args.base, device=args.device)
    model.train()
    block_size = model.config.block_size
    print(f"[ft] base: {args.base} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"block {block_size})")

    with open(args.data) as f:
        docs = json.load(f)
    if args.identity:
        # ~10% identity docs, shuffled through the corpus
        idocs = [t.format(name=args.identity_name) for t in IDENTITY_TEMPLATES]
        docs = docs + idocs * max(1, len(docs) // (10 * len(idocs)))
    random.shuffle(docs)

    enc = tiktoken.get_encoding("gpt2")
    ids = []
    for d in docs:
        ids.extend(enc.encode_ordinary(d))
        ids.append(enc.eot_token)
    data_ids = torch.tensor(ids, dtype=torch.long)
    print(f"[ft] corpus: {len(docs)} docs, {len(data_ids):,} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, y = get_batch(data_ids, block_size, args.batch_size, args.device, torch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it == 1 or it % 20 == 0:
            print(f"[ft] iter {it}/{args.iters}  loss {loss.item():.3f}  "
                  f"({(time.time() - t0)/it:.1f}s/iter)")

    model.eval()
    out_ckpt = {
        "model": model.state_dict(),
        "model_args": ckpt["model_args"],
        "iter_num": ckpt.get("iter_num", 0) + args.iters,
        "best_val_loss": loss.detach(),
        "config": ckpt.get("config", {}),
        "finetuned_from": os.path.basename(args.base),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out_ckpt, args.out)
    print(f"[ft] saved -> {args.out} ({os.path.getsize(args.out)/1e6:.0f}MB, "
          f"final loss {loss.item():.3f})")


if __name__ == "__main__":
    main()
