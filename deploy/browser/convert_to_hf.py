#!/usr/bin/env python3
"""
Convert an m.ai nanoGPT checkpoint -> HuggingFace GPT2LMHeadModel -> (ready for
ONNX export). nanoGPT was built to mirror GPT-2's naming, so the only real work
is transposing the four Linear weights per block that HF stores as Conv1D.

    python deploy/browser/convert_to_hf.py --ckpt checkpoints/potus-chat-1.0.pt \
        --out deploy/browser/hf/potus-chat

Then export ONNX:
    optimum-cli export onnx --model deploy/browser/hf/potus-chat \
        --task text-generation-with-past deploy/browser/onnx/potus-chat

The script also runs a NUMERICAL CHECK: it compares the HF model's logits
against the original nanoGPT model on a random input. If max abs diff is tiny,
the conversion is faithful and the browser will behave like the server.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "mai"))

# the 4 matrices HF keeps as Conv1D ([in,out]) but nanoGPT keeps as Linear ([out,in])
TRANSPOSE = ("attn.c_attn.weight", "attn.c_proj.weight",
             "mlp.c_fc.weight", "mlp.c_proj.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from model import load_checkpoint

    nano, ckpt = load_checkpoint(args.ckpt, device="cpu")
    nano.eval()
    a = ckpt["model_args"]
    print(f"[convert] {args.ckpt}: {a['n_layer']}L {a['n_head']}H {a['n_embd']}d "
          f"block={a['block_size']} vocab={a['vocab_size']}")

    # the gpt2 tokenizer knows 50257 tokens; the model pads to 50304. Those 47
    # rows are never produced by real text — slice them so the browser's gpt2
    # tokenizer is an exact match and can decode every token the model emits.
    GPT2_VOCAB = 50257

    cfg = GPT2Config(
        vocab_size=GPT2_VOCAB, n_positions=a["block_size"],
        n_ctx=a["block_size"], n_embd=a["n_embd"], n_layer=a["n_layer"],
        n_head=a["n_head"], bos_token_id=50256, eos_token_id=50256,
        # nanoGPT uses exact erf GELU (nn.GELU); HF defaults to gelu_new (tanh
        # approx). Match nano or logits drift ~2e-3 and sampling diverges.
        activation_function="gelu",
    )
    hf = GPT2LMHeadModel(cfg)

    nsd = nano.state_dict()
    hsd = hf.state_dict()
    new = {}
    missing = []
    for k in hsd:
        # HF has attn.bias / attn.masked_bias buffers nanoGPT doesn't — keep HF's
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            new[k] = hsd[k]
            continue
        src = k
        if src not in nsd:
            missing.append(k)
            new[k] = hsd[k]
            continue
        w = nsd[src]
        if any(k.endswith(t) for t in TRANSPOSE):
            w = w.t().contiguous()      # Linear [out,in] -> Conv1D [in,out]
        if k in ("transformer.wte.weight", "lm_head.weight"):
            w = w[:GPT2_VOCAB].contiguous()   # drop the 47 padding rows
        new[k] = w
    if missing:
        print(f"[convert] warn: {len(missing)} HF keys not in nano (kept HF init): "
              f"{missing[:4]}{'…' if len(missing) > 4 else ''}")
    hf.load_state_dict(new, strict=True)
    hf.eval()

    # ── numerical faithfulness check ──
    torch.manual_seed(0)
    ids = torch.randint(0, 50257, (1, 32))
    with torch.no_grad():
        # nano returns only last-position logits at inference; ask for full via targets path
        nano_logits = nano.lm_head(_nano_hidden(nano, ids))[..., :50257]
        hf_logits = hf(ids).logits
    diff = (nano_logits - hf_logits).abs().max().item()
    print(f"[convert] max|Δlogits| nano vs HF = {diff:.2e}  "
          f"({'OK — faithful' if diff < 1e-3 else 'TOO BIG — conversion wrong!'})")
    if diff >= 1e-3:
        sys.exit("conversion failed numerical check")

    os.makedirs(args.out, exist_ok=True)
    hf.save_pretrained(args.out)
    # ship a gpt2 tokenizer alongside so transformers.js / optimum find it
    from transformers import GPT2TokenizerFast
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(args.out)
    print(f"[convert] saved HF model + tokenizer -> {args.out}")


def _nano_hidden(nano, idx):
    """Recompute nanoGPT's pre-lm_head hidden states for ALL positions
    (model.forward only returns the last one at inference)."""
    import torch
    tconf = nano.transformer
    pos = torch.arange(idx.size(1))
    x = tconf.drop(tconf.wte(idx) + tconf.wpe(pos))
    for blk in tconf.h:
        x = blk(x)
    return tconf.ln_f(x)


if __name__ == "__main__":
    main()
