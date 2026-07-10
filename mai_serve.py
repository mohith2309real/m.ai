#!/usr/bin/env python3
"""
m.ai serve — an OpenAI-compatible API server for the potato registry.

Point ANY OpenAI-compatible client at it — including Model Deck
(backend: OpenAI-compatible, URL http://localhost:8123) — and chat with
your own from-scratch weights.

    python mai_serve.py                # serves every available registry model
    python mai_serve.py --port 8123

Endpoints: GET /v1/models, POST /v1/chat/completions (stream + non-stream).
The prompt sent to the potato is the last user message, raw — these are
base models, not chat models; manage your expectations and your dreams.
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "mai"))

_cache = {}
_cache_lock = threading.Lock()


def registry():
    with open(os.path.join(REPO, "models", "registry.json")) as f:
        return json.load(f)["models"]


def available_models():
    out = {}
    for name, m in registry().items():
        if "alias_of" in m:
            continue
        path = os.path.join(REPO, m["checkpoint"])
        if m.get("status") == "available" and os.path.exists(path) \
                and os.path.getsize(path) > 1000:
            out[name] = m
    return out


def get_model(name, spec):
    with _cache_lock:
        if name not in _cache:
            from model import load_checkpoint
            print(f"[serve] loading {name} <- {spec['checkpoint']}")
            _cache.clear()  # one model in memory at a time; potatoes share a kitchen
            _cache[name] = load_checkpoint(os.path.join(REPO, spec["checkpoint"]))[0]
        return _cache[name]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print("[serve]", fmt % a)

    def _headers(self, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self._headers(204)

    def do_GET(self):
        if self.path == "/v1/models":
            self._headers()
            data = [{"id": n, "object": "model", "owned_by": "m.ai"}
                    for n in available_models()]
            self.wfile.write(json.dumps({"object": "list", "data": data}).encode())
        else:
            self._headers(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._headers(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        models = available_models()
        name = body.get("model")
        if name not in models:
            self._headers(404)
            self.wfile.write(json.dumps({"error": {"message":
                f"unknown/unavailable model '{name}' — GET /v1/models"}}).encode())
            return
        spec = models[name]
        prompt = next((m["content"] for m in reversed(body.get("messages", []))
                       if m.get("role") == "user"), "Once upon a time")

        import tiktoken
        import torch
        enc = tiktoken.get_encoding("gpt2")
        model = get_model(name, spec)
        prof = spec["profile"]
        temperature = float(body.get("temperature") or prof["temperature"])
        max_toks = min(int(body.get("max_tokens") or prof["max_new_tokens"]), 500)
        idx = torch.tensor([enc.encode_ordinary(prompt)[-model.config.block_size + 1:]],
                           dtype=torch.long)

        if body.get("stream"):
            self._headers(200, "text/event-stream")
            gen_ids, sent = [], ""
            with torch.no_grad():
                for _ in range(max_toks):
                    cond = idx[:, -model.config.block_size:]
                    logits = model(cond)[:, -1, :] / max(temperature, 1e-5)
                    v, _ = torch.topk(logits, prof["top_k"])
                    logits[logits < v[:, [-1]]] = -float("inf")
                    nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                    idx = torch.cat((idx, nxt), dim=1)
                    # decode cumulatively so multi-byte UTF-8 split across BPE
                    # tokens never emits replacement characters
                    gen_ids.append(nxt.item())
                    full = enc.decode(gen_ids)
                    if full.endswith("\ufffd"):
                        continue  # mid-character; wait for the next token
                    tok, sent = full[len(sent):], full
                    chunk = {"object": "chat.completion.chunk", "model": name,
                             "choices": [{"index": 0, "delta": {"content": tok},
                                          "finish_reason": None}]}
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = model.generate(idx, max_toks, temperature=temperature,
                                 top_k=prof["top_k"])
            text = enc.decode(out[0, idx.size(1):].tolist())
            self._headers()
            self.wfile.write(json.dumps({
                "object": "chat.completion", "model": name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
            }).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    names = list(available_models())
    print(f"[serve] m.ai potato server on http://localhost:{args.port}/v1")
    print(f"[serve] serving: {', '.join(names) or 'NOTHING — git lfs pull first'}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
