#!/usr/bin/env python3
"""
Mythos — chat client for OUR OWN models. No external labs.

Talks to any OpenAI-compatible server; the default is m.ai's own potato
server (python mai_serve.py). We make our own models here.

Usage:
    python mai_serve.py &                          # serve the registry
    python mythos/mythos_client.py                 # interactive chat
    python mythos/mythos_client.py "a question"    # one-shot

Env:
    MYTHOS_ENGINE      model id (default: ropus-0.5; falls back to
                       ropus-0.3 if not available yet)
    MYTHOS_BASE_URL    default http://localhost:8123/v1 (mai_serve)
    MOONSHOT_API_KEY   only used as a bearer token for servers that want
                       auth; not needed for m.ai's own server
"""
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persona import MYTHOS_SYSTEM_PROMPT

BASE_URL = os.environ.get("MYTHOS_BASE_URL", "http://localhost:8123/v1").rstrip("/")
API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
ENGINE = os.environ.get("MYTHOS_ENGINE", "ropus-0.5")
FALLBACK_ENGINE = "ropus-0.3"


def _post(path, payload):
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def chat(messages, model=None, temperature=0.6):
    """One chat completion. Returns assistant text. Falls back to
    FALLBACK_ENGINE if the requested model id doesn't exist yet.
    The Mythos persona is prepended only when the caller didn't supply
    its own system message (the distiller supplies a clean one)."""
    model = model or ENGINE
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": MYTHOS_SYSTEM_PROMPT}] + messages
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    try:
        resp = _post("/chat/completions", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code in (400, 404) and model != FALLBACK_ENGINE and "model" in body.lower():
            print(f"[mythos] engine '{model}' not available, falling back to "
                  f"'{FALLBACK_ENGINE}'", file=sys.stderr)
            return chat(messages, model=FALLBACK_ENGINE, temperature=temperature)
        raise SystemExit(f"[mythos] API error {e.code}: {body[:300]}")
    return resp["choices"][0]["message"]["content"]


def main():
    if len(sys.argv) > 1:
        print(chat([{"role": "user", "content": " ".join(sys.argv[1:])}]))
        return

    print(f"👑 Mythos on {ENGINE} (100% m.ai weights) — Ctrl-C or 'exit' to leave")
    history = []
    while True:
        try:
            user = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[mythos] the org thanks you for your engagement")
            break
        if not user or user.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user})
        answer = chat(history)
        history.append({"role": "assistant", "content": answer})
        print(f"\nmythos > {answer}")


if __name__ == "__main__":
    main()
