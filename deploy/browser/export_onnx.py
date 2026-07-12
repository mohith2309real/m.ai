#!/usr/bin/env python3
"""
Export a converted HF GPT2 model (from convert_to_hf.py) to a plain ONNX graph:
    input_ids [1, seq]  ->  logits [1, seq, vocab]
No KV-cache — the browser re-runs the whole (<= block_size) context each token,
which is trivial at 16-51M params and keeps the graph and JS dead simple.

    python deploy/browser/export_onnx.py --hf deploy/browser/hf/potus-chat \
        --out deploy/browser/web/potus-chat/model.onnx

Validates the ONNX against the torch model before saving is considered done.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(args.hf)
    model.eval()

    class LogitsOnly(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids):
            return self.m(input_ids=input_ids).logits

    wrapped = LogitsOnly(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dummy = torch.randint(0, 50257, (1, 16))
    torch.onnx.export(
        wrapped, (dummy,), args.out,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=14, do_constant_folding=True,
    )

    # ── validate ONNX vs torch on fresh random input ──
    ids = torch.randint(0, 50257, (1, 24))
    with torch.no_grad():
        t_logits = wrapped(ids).numpy()
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    o_logits = sess.run(["logits"], {"input_ids": ids.numpy().astype(np.int64)})[0]
    diff = float(np.abs(t_logits - o_logits).max())
    if diff >= 1e-3:
        sys.exit(f"onnx export failed numerical check ({diff:.2e})")

    # collapse external-data (.onnx.data) into ONE self-contained file so the
    # browser only fetches a single URL
    import onnx
    m = onnx.load(args.out)
    onnx.save_model(m, args.out, save_as_external_data=False)
    data = args.out + ".data"
    if os.path.exists(data):
        os.remove(data)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[onnx] {args.out}  ({mb:.0f}MB single file)  "
          f"max|Δ| torch vs onnx = {diff:.2e}  ({'OK' if diff < 1e-3 else 'MISMATCH!'})")


if __name__ == "__main__":
    main()
