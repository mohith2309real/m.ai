#!/usr/bin/env python3
"""
ALUS 0.2 — ALL-8-CORES TPU trainer (data-parallel via xmp.spawn).

Run as a SCRIPT from a Kaggle cell (NOT inline — multiprocessing needs a real
process), after uploading this file or cloning the repo:

    # text pretraining, all 8 cores, resumes from the checkpoint:
    !python train_alus_8core.py --phase text  --steps 30000

    # chat fine-tune on the rich corpus (build it first with build_chat_corpus.py):
    !python train_alus_8core.py --phase chat  --steps 6000 --data /kaggle/working/chat_big.json

Each of the 8 TPU cores runs a replica on different data; gradients are
all-reduced every step (xm.optimizer_step), so it's one big effective batch
(per_core_batch * 8) and ~8x the throughput of the single-core cell.

Checkpoints (master core only) land in /kaggle/working/alus02 in m.ai format:
    --phase text -> alus-0.2-text.pt      (base brain, continues text)
    --phase chat -> alus-0.2-chat.pt      (chat: true — replies and stops)
Both resume automatically: rerun the same command after a session dies.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

OUT = "/kaggle/working/alus02"
USER_TAG, BOT_TAG, EOT = "<|user|>", "<|mai|>", 50256

# 406M-class config (matches the single-core cell so checkpoints interchange)
CFG = dict(n_layer=24, n_head=16, n_embd=1024, block_size=512,
           vocab_size=50304, dropout=0.0, bias=True)


# ─────────────────────────── model (m.ai nanoGPT) ───────────────────────────
class SelfAttn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.c_attn = nn.Linear(d, 3*d); s.c_proj = nn.Linear(d, d); s.h, s.d = h, d
    def forward(s, x):
        B, T, E = x.size()
        q, k, v = s.c_attn(x).split(s.d, dim=2)
        q, k, v = (t.view(B, T, s.h, E//s.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.c_proj(y.transpose(1, 2).contiguous().view(B, T, E))


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.c_fc = nn.Linear(d, 4*d); s.gelu = nn.GELU(); s.c_proj = nn.Linear(4*d, d)
    def forward(s, x): return s.c_proj(s.gelu(s.c_fc(x)))


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.ln_1, s.attn, s.ln_2, s.mlp = nn.LayerNorm(d), SelfAttn(d, h), nn.LayerNorm(d), MLP(d)
    def forward(s, x):
        x = x + s.attn(s.ln_1(x)); return x + s.mlp(s.ln_2(x))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c["vocab_size"], c["n_embd"]),
            wpe=nn.Embedding(c["block_size"], c["n_embd"]),
            h=nn.ModuleList([Block(c["n_embd"], c["n_head"]) for _ in range(c["n_layer"])]),
            ln_f=nn.LayerNorm(c["n_embd"])))
        s.lm_head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False)
        s.transformer.wte.weight = s.lm_head.weight
        s.apply(lambda m: isinstance(m, nn.Linear) and nn.init.normal_(m.weight, std=0.02))
    def forward(s, idx, targets=None):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = s.transformer.wte(idx) + s.transformer.wpe(pos)
        for b in s.transformer.h: x = b(x)
        logits = s.lm_head(s.transformer.ln_f(x))
        if targets is None: return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def strip(sd): return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


# ─────────────────────────── data ───────────────────────────
def text_stream(index):
    """Per-core FineWeb-Edu stream — each core sees a different shuffle so the
    8 replicas train on different data (real data parallelism)."""
    from datasets import load_dataset
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train",
                      streaming=True).shuffle(seed=1234 + index, buffer_size=10000)
    it = iter(ds)
    def refill():
        toks = []
        while len(toks) < 2_000_000:
            toks.extend(enc.encode_ordinary(next(it)["text"]) + [EOT])
        return torch.tensor(toks, dtype=torch.long)
    return refill


def chat_stream(path):
    """Rich chat corpus -> (ids, mask) with loss masking. Same for all cores;
    each core samples different batches."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    convos = json.load(open(path))
    random.shuffle(convos)
    ids, mask = [], []
    for c in convos:
        row_ids, row_mask = [], []
        ok = True
        for m in c["messages"]:
            if m["role"] == "user":
                t = enc.encode_ordinary(f"{USER_TAG}\n{m['content']}\n"); row_ids += t; row_mask += [0]*len(t)
            else:
                tag = enc.encode_ordinary(f"{BOT_TAG}\n"); row_ids += tag; row_mask += [0]*len(tag)
                rep = enc.encode_ordinary(m["content"]) + [EOT]; row_ids += rep; row_mask += [1]*len(rep)
        if len(row_ids) <= CFG["block_size"]:
            ids += row_ids; mask += row_mask
    return torch.tensor(ids), torch.tensor(mask)


# ─────────────────────────── per-core training fn ───────────────────────────
def _run(index, args):
    dev = xm.xla_device()
    torch.manual_seed(1337 + index); random.seed(1337 + index)
    is_chat = args.phase == "chat"
    ckpt_path = os.path.join(OUT, "alus-0.2-chat.pt" if is_chat else "alus-0.2-text.pt")
    os.makedirs(OUT, exist_ok=True)

    model = GPT(CFG).to(dev)
    start = 0
    # resume: chat starts from the text brain; text resumes itself
    seed_from = ckpt_path if os.path.exists(ckpt_path) else os.path.join(OUT, "alus-0.2-text.pt")
    if os.path.exists(seed_from):
        sd = torch.load(seed_from, map_location="cpu")
        model.load_state_dict(strip(sd["model"]))
        if seed_from == ckpt_path:
            start = sd.get("iter_num", 0)
        xm.master_print(f"[{args.phase}] seeded from {os.path.basename(seed_from)} "
                        f"(resume step {start})")
    model = model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    B, block = args.batch, CFG["block_size"]

    if is_chat:
        cids, cmask = chat_stream(args.data)
        def get_batch():
            ix = torch.randint(len(cids) - block - 1, (B,))
            x = torch.stack([cids[i:i+block] for i in ix])
            y = torch.stack([cids[i+1:i+1+block] for i in ix]).clone()
            mb = torch.stack([cmask[i+1:i+1+block] for i in ix]); y[mb == 0] = -1
            return x.to(dev), y.to(dev)
    else:
        refill = text_stream(index); buf = refill()
        def get_batch():
            ix = torch.randint(len(buf) - block - 1, (B,))
            x = torch.stack([buf[i:i+block] for i in ix])
            y = torch.stack([buf[i+1:i+1+block] for i in ix])
            return x.to(dev), y.to(dev)

    def lr_at(it):
        if it < args.warmup: return args.lr * it / max(1, args.warmup)
        p = min((it - args.warmup) / max(1, args.steps - args.warmup), 1.0)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    def save(step, loss_val):
        bl = loss_val.item() if hasattr(loss_val, "item") else float(loss_val)
        state = {"model": {k: v.cpu() for k, v in model.state_dict().items()},
                 "model_args": CFG, "iter_num": step, "best_val_loss": bl, "config": {}}
        if is_chat: state["chat"] = True
        # single process: plain torch.save (xm.save's rendezvous needs a mesh)
        (torch.save if args.single else xm.save)(state, ckpt_path)
        xm.master_print(f"[ckpt] {ckpt_path} @ step {step}")

    model.train(); t0 = time.time(); done = start
    while done < args.steps:
        for g in opt.param_groups: g["lr"] = lr_at(done)
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        if args.single:
            opt.step(); xm.mark_step()     # one core: plain step (the proven path)
        else:
            xm.optimizer_step(opt)         # all-reduce grads across the 8 cores + step
        done += 1
        if done % 20 == 0:
            xm.add_step_closure(lambda l, d: xm.master_print(
                f"[{args.phase}] step {d}/{args.steps}  loss {l.item():.3f}  "
                f"({(time.time()-t0)/(d-start):.2f}s/step, "
                f"{'1 core' if args.single else '8 cores'})"),
                args=(loss, done))
        if not is_chat and done % 2000 == 0:
            buf = refill()
        if done % args.save_every == 0:
            save(done, loss.detach())
    save(done, loss.detach())
    xm.master_print(f"[{args.phase}] DONE -> {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["text", "chat"], required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8, help="PER-CORE batch (x8 cores effective)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--data", default="/kaggle/working/chat_big.json")
    ap.add_argument("--single", action="store_true",
                    help="one TPU core, no xmp.spawn — reliable on Kaggle v5e")
    args = ap.parse_args()
    if args.phase == "chat" and args.lr == 3e-4:
        args.lr = 3e-5                     # gentle fine-tune LR
    if args.single:
        _run(0, args)                      # single core — the proven-reliable path
    else:
        xmp.spawn(_run, args=(args,))      # all 8 cores (fragile on some Kaggle TPUs)


if __name__ == "__main__":
    main()
