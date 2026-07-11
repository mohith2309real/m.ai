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
    /config               view/set persisted settings (model, temperature, theater)
    /status               server, model, uptime — the org dashboard
    /cost                 what this session cost you
    /credits              who built this (mohith. it was mohith)
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
import shutil
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
    print(c(DIM, "  " + BANNER_TAG))
    width = min(shutil.get_terminal_size((80, 24)).columns, 100)
    print(c(DIM, "m.ai can make mistakes. it is literally a potato.".rjust(width)) + "\n")


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


def chat(model, history, on_token, max_tokens=None, temperature=None):
    payload = {"model": model, "messages": history, "stream": True}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
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


# ── ceremony ───────────────────────────────────────────────────────────
CREDITS = [
    "m.ai — a very real org",
    "designed & developed by mohith",
    "models: potus · ropus · alus — trained from scratch",
    "© 2026 m.ai · property of m.ai",
    "AI can make mistakes. ours makes more. it is a potato.",
]


def print_credits():
    """The org's legal department (also mohith) insists on this."""
    w = max(len(s) for s in CREDITS)
    print(c(VIOLET, "  ┌─" + "─" * w + "─┐"))
    for s in CREDITS:
        print(c(VIOLET, "  │ ") + s.ljust(w) + c(VIOLET, " │"))
    print(c(VIOLET, "  └─" + "─" * w + "─┘"))


def print_exit_footer():
    print(c(DIM, "© 2026 m.ai · designed & developed by mohith · property of m.ai"))
    print(c(DIM, "(the org thanks you for visiting the basement)"))


TAN = "38;5;180"
GREEN = "38;5;114"

PITTI_MOODS = {
    "idle": ("\u2500", "chief executive potato \u00b7 he is aware. he is fine."),
    "error": ("o", "pitti has seen the error. pitti is unbothered."),
}


def print_pitti(mood="idle"):
    """The org's mascot. He was here before the org."""
    mouth, note = PITTI_MOODS.get(mood, PITTI_MOODS["idle"])
    art = [
        (GREEN, "       \\|/"),
        (TAN,   "    .-~~~~~-."),
        (TAN,   "   /  \u2022   \u2022  \\"),
        (TAN,   "  |     " + mouth + "     |"),
        (TAN,   "   \\         /"),
        (TAN,   "    `-.___.-'  \u25e6"),
    ]
    label = ["", "", c(BOLD, "PITTI"), c(DIM, note),
             c(DIM, "(the \u25e6 is his pet rock)"), ""]
    for (shade, a), l in zip(art, label):
        print(c(shade, a.ljust(20)) + l)


# ── settings (the org has preferences now) ─────────────────────────────
CONFIG_PATH = os.path.expanduser("~/.mai.json")


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# maximum-effort keywords → theater scripts (longest/most specific first)
THINK_MODES = [
    ("ultracode", ["ULTRACODE ENGAGED — spawning 128 parallel potato agents…",
                   "…they are all the same potato",
                   "merging 128 identical opinions…"]),
    ("ultrathink", ["engaging ultrathink — allocating all 51 million neurons…",
                    "that is 0.006% of a Claude…",
                    "neurons allocated. quality not guaranteed."]),
    ("megathink", ["megathink: thinking medium-hard…",
                   "consulting the potato council…"]),
    ("think harder", ["thinking… harder…",
                      "furrowing the potato's brow…"]),
    ("max power", ["MAX POWER: redlining the potato…",
                   "cooling fans (we do not have fans)…"]),
]


def thinking_theater(msg):
    """Extended thinking, in the theatrical sense. Returns True if performed."""
    low = msg.lower()
    lines = next((ls for kw, ls in THINK_MODES if kw in low), None)
    if lines is None:
        return False
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    t0 = time.time()
    if sys.stdout.isatty():
        i = 0
        while time.time() - t0 < 2.2:
            status = lines[min(int((time.time() - t0) / 2.2 * len(lines)), len(lines) - 1)]
            sys.stdout.write("\r\033[2K" + c(TEAL, frames[i % len(frames)]) + " " + c(DIM, status))
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
    else:
        print(c(DIM, lines[0]))
    print(c(DIM, f"(thought for {time.time() - t0:.1f}s — mostly about starch)"))
    return True


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
    print_pitti()
    cfg = load_config()
    t0 = time.time()
    models = ensure_server()
    if model is None and cfg.get("model") in models:
        model = cfg["model"]
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
                print(c(DIM, "  psst: put 'ultrathink' or 'ultracode' in a message for maximum effort mode\n"))
            elif cmd == "credits":
                print_credits()
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
            elif cmd == "config":
                sub = arg.split()
                if not sub:
                    print(c(DIM, f"  config file: {CONFIG_PATH}"))
                    print(f"  model        {cfg.get('model', '(auto — best available)')}")
                    print(f"  temperature  {cfg.get('temperature', '(model default)')}")
                    print(f"  theater      {'on' if cfg.get('theater', True) else 'off'}")
                    print(c(DIM, "  /config set model|temperature|theater <value>"))
                elif sub[0] == "set" and len(sub) >= 3:
                    key, val = sub[1], " ".join(sub[2:])
                    if key == "model":
                        if val in ensure_server():
                            cfg["model"] = val
                            model = val
                            save_config(cfg)
                            print(c(TEAL, f"[mai] default model → {val} (saved)"))
                        else:
                            print(c(RED, f"[mai] unknown model '{val}' — /models"))
                    elif key == "temperature":
                        try:
                            if val in ("default", "unset"):
                                cfg.pop("temperature", None)
                            else:
                                cfg["temperature"] = float(val)
                            save_config(cfg)
                            print(c(TEAL, f"[mai] temperature → {cfg.get('temperature', 'model default')}"))
                        except ValueError:
                            print(c(RED, "[mai] temperature wants a number, or 'default'"))
                    elif key == "theater":
                        cfg["theater"] = val.lower() in ("on", "true", "1", "yes")
                        save_config(cfg)
                        print(c(TEAL, "[mai] thinking theater "
                                + ("on (maximum effort restored)" if cfg["theater"]
                                   else "off (the org will think quietly)")))
                    else:
                        print(c(RED, f"[mai] unknown setting '{key}' — model, temperature, theater"))
                else:
                    print(c(RED, "[mai] usage: /config  |  /config set <key> <value>"))
            elif cmd == "status":
                up = int(time.time() - t0)
                turns = sum(1 for m in history if m["role"] == "user")
                ver = "?"
                vp = os.path.join(REPO, "VERSION")
                if os.path.exists(vp):
                    ver = open(vp).read().strip()
                print(c(BOLD, "  m.ai status") + c(DIM, " — the org dashboard"))
                print(f"  server   {SERVER}  ({len(ensure_server())} models)")
                print(f"  model    {model}")
                print(f"  version  {ver}")
                print(f"  uptime   {up // 60}m {up % 60}s")
                print(f"  turns    {turns}")
                print(c(DIM, f"  config   {CONFIG_PATH}"))
            elif cmd == "cost":
                print("  session cost:   ₹0.00")
                print("  lifetime cost:  ₹0.00")
                print(c(DIM, "  the org cannot afford costs. that is the business model."))
            else:
                print(c(RED, f"[mai] unknown command /{cmd} — /help"))
            continue

        # regular chat turn
        history.append({"role": "user", "content": user})
        boosted = cfg.get("theater", True) and thinking_theater(user)  # maximum effort, minimum result
        print(c(BOLD, f"{model} "), end="", flush=True)
        try:
            reply = chat(model, history, lambda t: print(t, end="", flush=True),
                         max_tokens=400 if boosted else None,
                         temperature=cfg.get("temperature"))
        except (urllib.error.URLError, OSError) as e:
            print(c(RED, f"\n[mai] server error: {e}"))
            print_pitti("error")
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": reply})
        maybe_run_model_tools(reply, history)

    print_exit_footer()  # both /exit and ctrl-d/ctrl-c land here


if __name__ == "__main__":
    main()
