"""
m.ai on Kaggle — a free, public potato server in ~4 cells.

Gives you a real https://<random>.trycloudflare.com URL backed by a free
Kaggle CPU session. No account needed for the tunnel; you just need a
(free) Kaggle account with Internet enabled on the notebook.

SETUP (once):
  - kaggle.com -> New Notebook
  - right sidebar -> Settings -> Internet: ON  (needs phone verification once)
  - paste the four cells below into the notebook, run top to bottom

The weights come from the free HF model repo (mohith2309/mai-chat-checkpoints),
NOT from GitHub LFS — so the clone is instant and there's no bandwidth cap.

Keep the LAST cell running: it blocks to hold the tunnel open. When the
session ends (~9h or idle), just re-run — you'll get a NEW url each time,
so repoint your website's DEFAULT_API / the Deck's server field to it.

────────────────────────────────────────────────────────────────────────
CELL 1 — code + deps (skip LFS: real weights arrive in cell 2)
────────────────────────────────────────────────────────────────────────
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mohith2309/m.ai.git /kaggle/working/mai
!pip -q install torch tiktoken huggingface_hub

────────────────────────────────────────────────────────────────────────
CELL 2 — the 3 chat checkpoints from the free HF model repo
────────────────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('/kaggle/working/mai/checkpoints', exist_ok=True)
for f in ['potus-chat-1.0.pt', 'ropus-chat-2.5.pt', 'alus-chat-1.5.pt']:
    p = hf_hub_download('mohith2309/mai-chat-checkpoints', f)
    shutil.copy(p, f'/kaggle/working/mai/checkpoints/{f}')
    print('got', f)

────────────────────────────────────────────────────────────────────────
CELL 3 — start the potato server on 0.0.0.0:8123 (CORS open so a website can reach it)
────────────────────────────────────────────────────────────────────────
import subprocess, os, time
env = dict(os.environ, MAI_HOST='0.0.0.0', MAI_ORIGINS='*')
srv = subprocess.Popen(['python', 'mai_serve.py', '--host', '0.0.0.0', '--port', '8123'],
                       cwd='/kaggle/working/mai', env=env)
time.sleep(6)
!curl -s http://localhost:8123/v1/models

────────────────────────────────────────────────────────────────────────
CELL 4 — cloudflared quick tunnel, prints the PUBLIC URL (keep this running)
────────────────────────────────────────────────────────────────────────
import subprocess, re
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared && chmod +x /tmp/cloudflared
proc = subprocess.Popen(['/tmp/cloudflared', 'tunnel', '--url', 'http://localhost:8123'],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='')
    m = re.search(r'https://[-\w]+\.trycloudflare\.com', line)
    if m:
        print('\n\n🥔  PUBLIC URL:', m.group(0))
        print('   open that url  -> the Deck')
        print('   put it in the website Settings / DEFAULT_API to wire the login site to it')
# (loop keeps running to hold the tunnel open — do not stop this cell)
"""
