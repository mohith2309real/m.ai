#!/usr/bin/env python3
"""
m.ai serve — an OpenAI-compatible API server for the potato registry.

Point ANY OpenAI-compatible client at it — including Model Deck
(backend: OpenAI-compatible, URL http://localhost:8123) — and chat with
your own from-scratch weights.

    python mai_serve.py                # serves every available registry model
    python mai_serve.py --port 8123

Endpoints: GET /v1/models, POST /v1/chat/completions (stream + non-stream).
Base models get the last user message, raw. Checkpoints saved with
"chat": True (train/finetune_chat_cpu.py) get the full history rendered
in the m.ai chat template — those actually reply instead of continuing.
"""
import argparse
import json
import os
import re
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


def generate_ids(model, idx, max_toks, temperature, top_k, penalty,
                 penalize_prompt=True):
    """Sample token ids one at a time with a repetition penalty.

    Small base models loop hard ("New York, New York, New York...") without
    one — CTRL-style: downweight every token already in the context.
    Chat models pass penalize_prompt=False: they must stay free to reuse the
    question's words ("capital", "France") — only their OWN output is
    penalized, else long conversations spiral into rare-token junk.
    """
    import torch
    seen = set(idx[0].tolist()) if penalize_prompt else set()
    seen.discard(50256)  # never penalize <|endoftext|> — stopping must stay easy
    with torch.no_grad():
        for _ in range(max_toks):
            cond = idx[:, -model.config.block_size:]
            logits = model(cond)[:, -1, :]
            if penalty > 1.0 and seen:
                ids = torch.tensor(sorted(seen), dtype=torch.long)
                vals = logits[0, ids]
                logits[0, ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
            logits = logits / max(temperature, 1e-5)
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
            tid = nxt.item()
            if tid == 50256:  # <|endoftext|> — the potato is done talking
                return
            seen.add(tid)
            yield tid


def get_model(name, spec):
    """Return (model, is_chat) — is_chat comes from the checkpoint itself
    (train/finetune_chat_cpu.py stamps "chat": True on what it saves)."""
    with _cache_lock:
        if name not in _cache:
            from model import load_checkpoint
            print(f"[serve] loading {name} <- {spec['checkpoint']}")
            _cache.clear()  # one model in memory at a time; potatoes share a kitchen
            model, ckpt = load_checkpoint(os.path.join(REPO, spec["checkpoint"]))
            _cache[name] = (model, bool(ckpt.get("chat")))
        return _cache[name]


STOP_MARKERS = ("<|user|>", "<|mai|>")
USER_TAG, BOT_TAG = STOP_MARKERS


def trim_stops(text):
    """(text, hit) — cut chat output at the first turn marker the model tries
    to hallucinate, and hold back a trailing partial marker so a stream never
    emits half of one."""
    hits = [text.find(s) for s in STOP_MARKERS if s in text]
    if hits:
        return text[:min(hits)], True
    for n in range(min(len(text), len(STOP_MARKERS[0]) - 1), 0, -1):
        tail = text[-n:]
        if any(s.startswith(tail) for s in STOP_MARKERS):
            return text[:-n], False
    return text, False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print("[serve]", fmt % a)

    def _cors_origin(self):
        # localhost pages only (any port) — a random website should not get to
        # use your potato. MAI_ORIGINS="*" restores the open-door policy.
        origin = self.headers.get("Origin", "")
        if os.environ.get("MAI_ORIGINS", "") == "*":
            return "*"
        if re.match(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$", origin):
            return origin
        return None

    def _headers(self, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Vary", "Origin")
        self.end_headers()

    def _bad_request(self, msg):
        self._headers(400)
        self.wfile.write(json.dumps({"error": {"message": msg}}).encode())

    def do_OPTIONS(self):
        self._headers(204)

    def do_GET(self):
        if self.path == "/v1/models":
            self._headers()
            data = [{"id": n, "object": "model", "owned_by": "m.ai"}
                    for n in available_models()]
            self.wfile.write(json.dumps({"object": "list", "data": data}).encode())
        elif self.path in ("/", "/index.html"):
            # single-port mode: the server hosts the m.ai Deck too
            ui = os.path.join(REPO, "interface", "index.html")
            if os.path.exists(ui):
                self._headers(200, "text/html; charset=utf-8")
                with open(ui, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._headers(404)
        elif self.path == "/brand/logo.svg":
            logo = os.path.join(REPO, "brand", "logo.svg")
            if os.path.exists(logo):
                self._headers(200, "image/svg+xml")
                with open(logo, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._headers(404)
        else:
            self._headers(404)

    def _authed(self):
        # opt-in: export MAI_API_KEY on the server to require it from clients
        want = os.environ.get("MAI_API_KEY", "")
        if not want:
            return True
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {want}"

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._headers(404)
            return
        if not self._authed():
            self._headers(401)
            self.wfile.write(json.dumps({"error": {"message":
                "MAI_API_KEY is set on this server — send Authorization: Bearer <key>"}}).encode())
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, json.JSONDecodeError):
            self._bad_request("invalid JSON body")
            return
        models = available_models()
        name = body.get("model")
        if name not in models:
            self._headers(404)
            self.wfile.write(json.dumps({"error": {"message":
                f"unknown/unavailable model '{name}' — GET /v1/models"}}).encode())
            return
        spec = models[name]

        import tiktoken
        import torch
        enc = tiktoken.get_encoding("gpt2")
        model, is_chat = get_model(name, spec)
        prof = spec["profile"]
        opts = body.get("options") or {}  # Ollama-style knobs from the Deck UI
        try:
            temperature = float(body.get("temperature") or opts.get("temperature")
                                or prof["temperature"])
            top_k = max(1, int(opts.get("top_k") or prof["top_k"]))
            penalty = float(body.get("repetition_penalty")
                            or opts.get("repeat_penalty")
                            or prof.get("repeat_penalty", 1.3))
            max_toks = min(int(body.get("max_tokens") or prof["max_new_tokens"]), 500)
        except (TypeError, ValueError):
            self._bad_request("temperature/top_k/max_tokens must be numbers")
            return
        msgs = body.get("messages", [])
        if is_chat:
            # render history in the template the model was tuned on — but the
            # context is TINY (256 tokens), so fit whole turns newest-first: a
            # chopped template is worse than a short memory
            budget = model.config.block_size - 48   # leave room for the reply
            tail = f"{BOT_TAG}\n"
            used = len(enc.encode(tail))
            parts = []
            for m in reversed(msgs):
                if m.get("role") == "user":
                    txt = f"{USER_TAG}\n{m.get('content', '')}\n"
                elif m.get("role") == "assistant":
                    txt = f"{BOT_TAG}\n{m.get('content', '')}<|endoftext|>"
                else:
                    continue
                toks = len(enc.encode(txt, allowed_special={"<|endoftext|>"}))
                if used + toks > budget and parts:
                    break                            # keep at least the newest turn
                parts.append(txt)
                used += toks
            ids = enc.encode("".join(reversed(parts)) + tail,
                             allowed_special={"<|endoftext|>"})
            ids = ids[-(model.config.block_size - 32):]
        else:
            prompt = next((m["content"] for m in reversed(msgs)
                           if m.get("role") == "user"), "Once upon a time")
            ids = enc.encode_ordinary(prompt)
        idx = torch.tensor([ids[-model.config.block_size + 1:]], dtype=torch.long)
        gen = generate_ids(model, idx, max_toks, temperature, top_k, penalty,
                           penalize_prompt=not is_chat)

        if body.get("stream"):
            self._headers(200, "text/event-stream")
            gen_ids, sent = [], ""
            for tid in gen:
                # decode cumulatively so multi-byte UTF-8 split across BPE
                # tokens never emits replacement characters
                gen_ids.append(tid)
                full = enc.decode(gen_ids)
                if full.endswith("\ufffd"):
                    continue  # mid-character; wait for the next token
                hit = False
                if is_chat:
                    full, hit = trim_stops(full)
                tok, sent = full[len(sent):], full
                if tok:
                    chunk = {"object": "chat.completion.chunk", "model": name,
                             "choices": [{"index": 0, "delta": {"content": tok},
                                          "finish_reason": None}]}
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
                if hit:
                    break  # model drifted into the next turn \u2014 its reply is over
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            text = enc.decode(list(gen))
            if is_chat:
                text = trim_stops(text)[0].strip()
            self._headers()
            self.wfile.write(json.dumps({
                "object": "chat.completion", "model": name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
            }).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default=os.environ.get("MAI_HOST", "127.0.0.1"),
                    help="bind address (default localhost-only; "
                         "--host 0.0.0.0 to serve your LAN)")
    args = ap.parse_args()
    names = list(available_models())
    print(f"[serve] m.ai potato server on http://localhost:{args.port}/v1")
    print(f"[serve] serving: {', '.join(names) or 'NOTHING — git lfs pull first'}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[serve] WARNING: bound to {args.host} — everyone on your network "
              "can use your potato")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
