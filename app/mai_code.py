#!/usr/bin/env python3
"""
mai code — m.ai's agentic terminal. Same shape as the big-lab CLIs:
a chat loop, slash commands, tools, and an update channel. The brain is
whatever the registry serves; the harness doesn't judge.

    python app/mai_code.py                 # picks the best available model
    python app/mai_code.py --model alus-1.0

Slash commands:
    /help                 this
    /models               list served models
    /model <name>         switch model
    /run <cmd>  (or !cmd) run a shell command, output goes into context
    /read <file>          read a file into context
    /ls [dir]             list a directory into context
    /clear                clear the conversation
    /update               pull the newest m.ai code + weights from GitHub
    /exit                 leave

Tool protocol (for models big enough to use it): if the model's reply
contains a line like `TOOL run: <cmd>` or `TOOL read: <path>`, mai code
asks you y/n and, if approved, executes it and feeds the result back.
Today's potatoes mostly tell stories instead of calling tools. The
harness is ready for the day they don't.

Zero dependencies beyond the repo (urllib + the m.ai server).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.environ.get("MAI_SERVER", "http://localhost:8123")

# ── tiny ANSI kit ──────────────────────────────────────────────────────
def c(code, s):
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s

VIOLET = "38;5;135"
TEAL = "38;5;43"
DIM = "2"
BOLD = "1"
RED = "38;5;204"

# the m.ai wordmark, violet→teal gradient, one color per line (like the logo)
BANNER_ART = [
    r"  ███╗   ███╗      █████╗ ██╗ ",
    r"  ████╗ ████║     ██╔══██╗██║ ",
    r"  ██╔████╔██║     ███████║██║ ",
    r"  ██║╚██╔╝██║     ██╔══██║██║ ",
    r"  ██║ ╚═╝ ██║ ██╗ ██║  ██║██║ ",
    r"  ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═╝╚═╝ ",
]
# violet → blue → teal, top to bottom (same gradient as the logo)
BANNER_SHADES = ["38;5;135", "38;5;99", "38;5;63", "38;5;39", "38;5;44", "38;5;43"]
BANNER_TAG = "c o d e  —  the org's agentic terminal · models from scratch · basement frontier"


def print_banner():
    for line, shade in zip(BANNER_ART, BANNER_SHADES):
        print(c(shade, line))
    print(c(DIM, "  " + BANNER_TAG) + "\n")


# ── server plumbing ────────────────────────────────────────────────────
def api(path, payload=None, timeout=300):
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_server():
    try:
        return [m["id"] for m in api("/v1/models")["data"]]
    except (urllib.error.URLError, OSError):
        print(c(DIM, f"[mai] server not running at {SERVER} — starting it..."))
        subprocess.Popen([sys.executable, os.path.join(REPO, "mai_serve.py")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(1)
            try:
                return [m["id"] for m in api("/v1/models")["data"]]
            except (urllib.error.URLError, OSError):
                pass
        raise SystemExit(c(RED, "[mai] could not start mai_serve.py — run it manually"))


def pick_default(models):
    for name in ("ropus-2.0", "alus-1.0", "ropus-0.5", "potus-0.5"):
        if name in models:
            return name
    return models[0] if models else None


def chat(model, history, on_token):
    payload = {"model": model, "messages": history, "stream": True}
    req = urllib.request.Request(SERVER + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                tok = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            text += tok
            on_token(tok)
    return text


# ── tools ──────────────────────────────────────────────────────────────
def tool_run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return (p.stdout + p.stderr).strip() or f"(exit {p.returncode}, no output)"


def tool_read(path):
    with open(os.path.expanduser(path)) as f:
        body = f.read()
    return body[:4000] + ("\n...(truncated)" if len(body) > 4000 else "")


def tool_ls(path="."):
    return "\n".join(sorted(os.listdir(os.path.expanduser(path) or ".")))


TOOL_RE = re.compile(r"^TOOL (run|read|ls): ?(.*)$", re.M)


def maybe_run_model_tools(reply, history):
    """If the model asked for a tool, confirm + execute + feed back."""
    for kind, arg in TOOL_RE.findall(reply)[:3]:
        ans = input(c(TEAL, f"[mai] model wants TOOL {kind}: {arg!r} — allow? [y/N] ")).strip().lower()
        if ans != "y":
            continue
        try:
            out = {"run": tool_run, "read": tool_read, "ls": tool_ls}[kind](arg)
        except Exception as e:  # tool errors go back to the model too
            out = f"(tool error: {e})"
        print(c(DIM, out))
        history.append({"role": "user", "content": f"TOOL RESULT ({kind} {arg}):\n{out}"})


def update():
    print(c(TEAL, "[mai] checking GitHub for updates..."))
    print(tool_run(f"git -C {REPO} fetch origin"))
    behind = tool_run(f"git -C {REPO} rev-list --count HEAD..@{{u}} 2>/dev/null || echo ?")
    if behind.strip() in ("0", ""):
        print(c(TEAL, "[mai] already up to date. The org ships fast but not that fast."))
        return
    print(c(TEAL, f"[mai] {behind.strip()} new commit(s) available."))
    if input("Pull code + weights now? [y/N] ").strip().lower() == "y":
        print(tool_run(f"git -C {REPO} pull --ff-only"))
        print(tool_run(f"git -C {REPO} lfs pull"))
        print(c(TEAL, "[mai] updated. Restart mai code to load new models."))


# ── main loop ──────────────────────────────────────────────────────────
def main():
    model = None
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    print_banner()
    models = ensure_server()
    model = model or pick_default(models)
    print(c(DIM, f"  server {SERVER} · {len(models)} models · talking to ") + c(BOLD, model))
    print(c(DIM, "  /help for commands\n"))

    system = "You are a helpful m.ai model inside the mai code terminal."
    sp = os.path.join(REPO, "mai", "SYSTEM_PROMPT.md")
    if os.path.exists(sp):
        system = open(sp).read().replace("{MODEL_NAME}", model)
    history = [{"role": "system", "content": system}]

    while True:
        try:
            user = input(c(VIOLET, "❯ ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user.startswith("!"):
            print(c(DIM, tool_run(user[1:])))
            history.append({"role": "user", "content": f"TOOL RESULT (run {user[1:]}):\n{tool_run(user[1:])[:2000]}"})
            continue
        if user.startswith("/"):
            cmd, _, arg = user[1:].partition(" ")
            if cmd == "exit":
                break
            elif cmd == "help":
                print(c(DIM, __doc__.split("Slash commands:")[1].split("Tool protocol")[0]))
            elif cmd == "models":
                for mname in ensure_server():
                    mark = c(TEAL, " ← current") if mname == model else ""
                    print(f"  {mname}{mark}")
            elif cmd == "model":
                if arg in ensure_server():
                    model = arg
                    print(c(TEAL, f"[mai] switched to {model}"))
                else:
                    print(c(RED, f"[mai] unknown model '{arg}' — /models"))
            elif cmd == "run":
                out = tool_run(arg)
                print(c(DIM, out))
                history.append({"role": "user", "content": f"TOOL RESULT (run {arg}):\n{out[:2000]}"})
            elif cmd == "read":
                try:
                    out = tool_read(arg)
                    print(c(DIM, out[:1500]))
                    history.append({"role": "user", "content": f"TOOL RESULT (read {arg}):\n{out}"})
                except OSError as e:
                    print(c(RED, f"[mai] {e}"))
            elif cmd == "ls":
                out = tool_ls(arg or ".")
                print(c(DIM, out))
                history.append({"role": "user", "content": f"TOOL RESULT (ls {arg or '.'}):\n{out}"})
            elif cmd == "clear":
                history = history[:1]
                print(c(TEAL, "[mai] context cleared"))
            elif cmd == "update":
                update()
            else:
                print(c(RED, f"[mai] unknown command /{cmd} — /help"))
            continue

        # regular chat turn
        history.append({"role": "user", "content": user})
        print(c(BOLD, f"{model} "), end="", flush=True)
        try:
            reply = chat(model, history, lambda t: print(t, end="", flush=True))
        except (urllib.error.URLError, OSError) as e:
            print(c(RED, f"\n[mai] server error: {e}"))
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": reply})
        maybe_run_model_tools(reply, history)


if __name__ == "__main__":
    main()
