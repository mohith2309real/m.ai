# Hosting m.ai for free

The m.ai Deck talks to its own origin (interface/index.html sets
`baseUrl = location.origin`), so once `mai_serve.py` runs on a host, the
Deck at `/` just works — no CORS, no config. Three ways to get it online.

---

## 1. Instant share, zero signup — Cloudflare Tunnel (runs from your Mac)

Up in 30 seconds, public URL, but only alive while your Mac and the tunnel
are running. Good for showing a friend right now.

```bash
brew install cloudflared
# with mai_serve already running on :8123 (the Deck preview does this):
cloudflared tunnel --url http://localhost:8123
```

It prints a `https://something.trycloudflare.com` URL. Share it. Ctrl-C ends it.

---

## 2. Persistent free host — Hugging Face Spaces  ⭐ recommended

Real URL, 16 GB RAM, no credit card. Sleeps after 48 h idle and wakes on
the next visit (~30 s cold start). Files are ready in `deploy/hf-space/`.

**Trim the weights first** (a public demo needs only the chat models, ~450 MB
instead of 1.8 GB — faster builds, well under quotas):

```bash
bash deploy/trim_for_deploy.sh   # stages a lean copy in deploy/_stage
```

**Then:**
1. Make a free account at huggingface.co (I can't create it for you).
2. New → Space → **Docker** (blank template) → name it `m-ai`.
3. Copy `deploy/hf-space/Dockerfile` and `deploy/hf-space/README.md` into
   the staged copy's root, then `git push` the staged copy to the Space:
   ```bash
   cd deploy/_stage
   cp ../hf-space/Dockerfile ../hf-space/README.md .
   git init && git lfs install && git lfs track "checkpoints/*.pt"
   git add -A && git commit -m "m.ai Deck on HF"
   git remote add space https://huggingface.co/spaces/<you>/m-ai
   git push space main
   ```
4. Watch it build (~5 min). Your Deck is live at
   `https://<you>-m-ai.hf.space`.

> Optional: set a `MAI_API_KEY` secret in Space settings if you don't want
> the world using your compute. Note GET endpoints (model list, Deck HTML)
> stay open even then — chat POSTs are what the key gates.

---

## 3. Always-on real VM — Oracle Cloud Always Free

24/7, 24 GB ARM, free forever — but requires a credit card at signup and
provisioning can throw capacity errors. Only worth it if you need it never
to sleep. Setup: provision an Ampere VM, `git clone`, `./launch.sh`, open
the firewall for the port, point a process manager (systemd) at
`mai_serve.py --host 0.0.0.0`.
