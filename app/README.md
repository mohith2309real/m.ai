# 🖥️ m.ai apps — the open-source Claude, in your terminal and browser

No cloud. No hosting bills. No API keys. Everything runs on your machine
against the from-scratch m.ai weights. This is what "open-source Claude"
means when you can't afford to host Claude.

## One command

```bash
git lfs pull            # get the weights (first time only)
./launch.sh             # server + web interface + agentic terminal
./launch.sh --no-tui    # skip the terminal, just server + web UI
```

## The pieces

| Piece | What | Run it |
|---|---|---|
| **mai_serve.py** | OpenAI-compatible API over the model registry | `python mai_serve.py` |
| **interface/** | m.ai Deck — the web chat UI (Model Deck's sibling) | serve + open in browser |
| **app/mai_code.py** | agentic terminal, same shape as the big-lab CLIs | `python app/mai_code.py` |

## mai code — the terminal agent

A chat loop with slash commands (`/models`, `/model`, `/read`, `/run`,
`/ls`, `/clear`, `/update`, `/exit`), `!cmd` shell passthrough, and a tool
protocol so a capable-enough model can ask to run commands and read files
(you approve each one). Today's potatoes prefer telling stories to calling
tools; the harness is ready for the models that won't.

## Updates — how the org ships without hosting anything

`/update` inside mai code (or `git pull && git lfs pull`) fetches the newest
code **and model weights** straight from GitHub. GitHub is the CDN. Releases
are git tags. Model upgrades are LFS pushes. Distribution costs the org ₹0,
which is exactly the budget.

`VERSION` tracks the current release; the app compares your checkout against
`origin` and tells you when you're behind.
