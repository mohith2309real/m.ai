#!/usr/bin/env python3
"""
Turn a base potato into a *chat* potato. Same basement-tier CPU fine-tune
as finetune_cpu.py, but on conversations instead of flat text — and with
loss masking, so the model learns to PRODUCE the reply, not to parrot the
user's question back.

    python train/finetune_chat_cpu.py \
        --base checkpoints/ropus-2.0.pt \
        --data data/chat.json --identity-name Ropus \
        --iters 400 --out checkpoints/ropus-chat-2.5.pt

--data is the JSON from prepare_chat_data.py (list of {"messages":[...]}).
Identity conversations (who are you / who made you) are mixed in ~12% so
the chat model still knows it is an m.ai model.

Chat template (plain gpt2 BPE):
    <|user|>\\n{user}\\n<|mai|>\\n{assistant}<|endoftext|>
Only the assistant span (+ its trailing <|endoftext|>) contributes to the
loss; the prompt tokens are masked to -1 (the model's cross-entropy uses
ignore_index=-1). That is the whole trick behind "receptive".
"""
import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mai"))

# identity, but as CONVERSATIONS so it lands in chat format
IDENTITY_CONVOS = [
    ("Who are you?",
     "I am {name}, a model made by m.ai. m.ai is a small independent lab that "
     "trains its own models from scratch. I am here to help you."),
    ("Who made you?",
     "I was made by m.ai — a tiny lab that builds its own models (Potus, Ropus "
     "and Alus) from scratch. I am {name}."),
    ("What can you do?",
     "I am {name}, an m.ai model. I can chat, answer questions, and help you "
     "think things through. I am small and still learning, so I might be wrong "
     "sometimes — but I will always try."),
    ("What is m.ai?",
     "m.ai is the lab that made me. It trains language models from scratch — "
     "Potus, Ropus and Alus. I am {name}, one of them."),
    ("Are you ChatGPT?",
     "No — I am {name}, a model made by m.ai, trained from scratch. Different "
     "lab, much smaller, much more potato. But I am mine."),
    ("Hi!",
     "Hi! I am {name}, an m.ai model. What would you like to talk about?"),
]

USER_TAG = "<|user|>"
BOT_TAG = "<|mai|>"


def render(messages, enc, name):
    """Return (token_ids, mask) where mask=1 on tokens that count toward loss
    (the assistant reply + its <|endoftext|>), 0 on the masked prompt tokens."""
    ids, mask = [], []

    def add(text, learn):
        toks = enc.encode(text, allowed_special={"<|endoftext|>"})
        ids.extend(toks)
        mask.extend([1 if learn else 0] * len(toks))

    for m in messages:
        content = m["content"].replace("{name}", name)
        if m["role"] == "user":
            add(f"{USER_TAG}\n{content}\n", False)
        else:
            add(f"{BOT_TAG}\n", False)          # tag is prompt, not learned
            add(f"{content}", True)             # the reply IS the lesson
            add("<|endoftext|>", True)          # learn to stop
    return ids, mask


def build_stream(convos, enc, name, block_size):
    """Pack rendered conversations into one long (ids, targets) stream.
    targets = next-token ids, with masked positions set to -1 (ignored)."""
    ids_all, mask_all = [], []
    for c in convos:
        ids, mask = render(c["messages"], enc, name)
        if len(ids) > block_size:            # too long for this tiny context
            continue
        ids_all.extend(ids)
        mask_all.extend(mask)
    return ids_all, mask_all


def get_batch(ids, mask, block_size, batch_size, torch):
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([ids[i:i + block_size] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block_size] for i in ix]).clone()
    mb = torch.stack([mask[i + 1:i + 1 + block_size] for i in ix])
    y[mb == 0] = -1                          # ignore_index — mask the prompt
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="chat JSON from prepare_chat_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity-name", default="Ropus")
    ap.add_argument("--identity-frac", type=float, default=0.0,
                    help="fraction of m.ai identity convos to mix in (0 = pure chat)")
    ap.add_argument("--iters", type=int, default=400)
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
    print(f"[chat-ft] base {args.base} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M, block {block_size})")

    with open(args.data) as f:
        convos = json.load(f)

    # optionally mix in identity conversations (default off — pure chat first)
    if args.identity_frac > 0:
        idconvos = [{"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a}]}
                    for q, a in IDENTITY_CONVOS]
        n_id = max(len(idconvos), int(len(convos) * args.identity_frac))
        convos = convos + (idconvos * (n_id // len(idconvos) + 1))[:n_id]
        print(f"[chat-ft] mixed in {n_id} identity convos ({args.identity_name})")
    random.shuffle(convos)

    enc = tiktoken.get_encoding("gpt2")
    ids_list, mask_list = build_stream(convos, enc, args.identity_name, block_size)
    ids = torch.tensor(ids_list, dtype=torch.long)
    mask = torch.tensor(mask_list, dtype=torch.long)
    learned = int(mask.sum())
    print(f"[chat-ft] {len(convos)} convos -> {len(ids):,} tokens "
          f"({learned:,} learned / {len(ids)-learned:,} masked prompt)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, y = get_batch(ids, mask, block_size, args.batch_size, torch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it == 1 or it % 20 == 0:
            print(f"[chat-ft] iter {it}/{args.iters}  loss {loss.item():.3f}  "
                  f"({(time.time()-t0)/it:.1f}s/it)")

    model.eval()
    out = {
        "model": model.state_dict(),
        "model_args": ckpt["model_args"],
        "iter_num": ckpt.get("iter_num", 0) + args.iters,
        "best_val_loss": loss.detach(),
        "config": ckpt.get("config", {}),
        "finetuned_from": os.path.basename(args.base),
        "chat": True,                         # server reads this: render turns
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"[chat-ft] saved -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.0f}MB, final loss {loss.item():.3f})")


if __name__ == "__main__":
    main()
