# 🏢 m.ai — Definitely A Real AI Org™

> *m.ai (pronounced "em dot ay eye", formerly Pitti Rokini) is a frontier AI laboratory consisting of one (1) kid living in his mom's basement pretending to be an org. Our mission: beat Claude. Our budget: ₹0. Our confidence: unlimited.*

---

## 🚨 ANNOUNCING: MYTHOS — OUR MYTHOS-CLASS MODEL 🚨

Every real AI org has a model tier that sits above everything else. Anthropic has Mythos-class. So obviously **we also have Mythos-class now.** It's called **Mythos**, it's ours, and legal cannot stop us because we don't have legal.

**What Mythos actually is** (radical transparency, another thing real orgs pretend to have):

| Layer | What it is | Status |
|---|---|---|
| 🧠 **The Engine** | **Kimi K2.6** (1T-param MoE, open weights, made by an actual org) doing all the thinking via API | ✅ Works TODAY |
| 🎭 **The Identity** | The Mythos persona wrapped around it — our system prompt, our name on the door | ✅ Works TODAY |
| 🥔 **The Body** | Our own potato weights (350M, from scratch, nanoGPT) **distilled from Kimi K2.6's outputs** so the potato slowly learns to talk like a trillion parameters | 🔥 Pipeline ready — run it on Kaggle |

Is this what "Mythos-class" means? No. Is it what *our* Mythos-class means? Absolutely. We are an org. We define our own tiers.

```bash
# Talk to Mythos right now (needs a Moonshot API key, they have a free tier):
export MOONSHOT_API_KEY=sk-...
python mythos/mythos_client.py "explain recursion like I'm a potato"
```

---

## 🤔 What even is this repo?

The continuing adventures of increasingly unhinged language models trained **completely from scratch** — no fine-tuning*, no Llama, no shortcuts, no dignity — by a **kid with a MacBook, Claude Pro, and a concerning amount of free time.**

*\*okay FINE, Mythos uses distillation now. We grew as an org. Scaling laws humbled us. It happens to everyone.*

Built entirely in his mom's basement. 🏠 Now with **corporate branding.**

Powered by:
- 🆓 Kaggle free GPUs (30hrs/week, we use every single one)
- 🌙 Moonshot's Kimi K2.6 (they did the hard part, we did the vibes)
- 😭 Suffering (renewable resource)
- ☕ Whatever a kid drinks

---

## 🪨 Model Lineup (The Potato Family, now an Org Chart)

### ✅ *Pitti Rokini v0.1* (The OG Rock) — 16M params
Loss 10.81 → 1.94. It learned... something. ARC-AGI: 0% (certified rock behaviour).

### ✅ *Rokini Potoni* (The Rock-Potato) — 50M params
2x T4 DDP, 8 layers / 8 heads / 512 dim. Codes badly but yes. ARC-AGI: 0% (certified potato).

### 🔥 *Bigini Potoni* (The Big Potato) — 125M params
GPT-2 sized, dangerous levels of confidence. Still cooking. 👨‍🍳

### 👑 *MYTHOS* (The Mythos-Class Potato) — NEW FLAGSHIP
- **Engine mode:** Kimi K2.6 behind the Mythos persona — full trillion-param brain, available immediately
- **Potato mode:** 350M from-scratch model (12 layers / 16 heads / 1024 dim) trained on **data distilled from Kimi K2.6** — the potato inherits the wisdom of the mountain
- **ARC-AGI (engine mode):** actually decent for once (we take credit)
- **ARC-AGI (potato mode):** we'll get back to you (we won't)

---

## 🛠️ How to build Mythos yourself (the whole pipeline)

### Step 0 — Talk to Mythos immediately (engine mode)
```bash
export MOONSHOT_API_KEY=sk-...          # https://platform.moonshot.ai
python mythos/mythos_client.py           # interactive chat
python mythos/mythos_client.py "one-shot question"
# different engine version: export MYTHOS_ENGINE=kimi-k2.6
```
Runs on anything. It's an API call wearing a trench coat.

Got [Model Deck](https://github.com/mohith2309/yojana_mitra_app/tree/main/local_models) (our other product, we have products now)? Point its **OpenAI-compatible** backend at `https://api.moonshot.ai/v1` with your key — Mythos in a full chat UI.

### Step 1 — Distill the mountain into a dataset
```bash
python train/distill_dataset.py --samples 2000 --out data/mythos_distill.jsonl
```
Asks Kimi K2.6 thousands of questions (code, reasoning, stories — the potato curriculum), saves every answer. This is the "training data" a real org would pay millions for. We pay ₹0-per-token-until-the-free-tier-runs-out.

### Step 2 — Tokenize for nanoGPT
```bash
python train/prepare_nanogpt_data.py --in data/mythos_distill.jsonl --outdir data/mythos
```
tiktoken gpt2 BPE → `train.bin` / `val.bin`, exactly what our nanoGPT setup already eats.

### Step 3 — Train the potato (Kaggle, free tier, obviously)
```bash
torchrun --nproc_per_node=2 train.py config/phase_mythos_350M.py
```
350M params on 2x T4. Will it take the whole 30 hrs/week? Yes. Multiple weeks? Also yes. Is that a problem? We are an org, we have quarterly roadmaps now.

### Step 4 — Ship it locally
```bash
# after converting your checkpoint to GGUF (llama.cpp convert script):
ollama create mythos -f mythos/mythos.Modelfile
```

---

## 📊 The Benchmark Wall of Shame (Org Edition)

| Model | ARC-AGI | HumanEval | Notes |
|-------|---------|-----------|-------|
| Pitti Rokini v0.1 | 0% | 0% | It tried |
| Rokini Potoni | 0% | ~potato% | Still proud |
| Bigini Potoni | TBD | TBD | Cooking |
| **Mythos (engine)** | Kimi K2.6's score | Kimi K2.6's score | We simply take credit |
| **Mythos (potato)** | TBD | TBD | The distillation copium is real |
| Claude Opus 4.6 | high% | very high% | Our "goal" 💀 |

---

## 📁 Repo Structure

```
m.ai/                              ← rename the repo to this in Settings, we're an org now
├── README.md                      ← corporate communications (this)
├── mythos/
│   ├── mythos_client.py           ← Mythos engine mode: Kimi K2.6 + persona, works today
│   ├── persona.py                 ← the Mythos identity (our moat)
│   └── mythos.Modelfile           ← Ollama build for potato mode
├── train/
│   ├── distill_dataset.py         ← milk Kimi K2.6 for training data
│   └── prepare_nanogpt_data.py    ← jsonl → train.bin/val.bin (tiktoken gpt2)
├── config/
│   └── phase_mythos_350M.py       ← nanoGPT config for the Mythos potato
├── checkpoints/                   ← LFS pointers to 184MB of digital rock
└── benchmarks/                    ← a list of zeros (growing!)
```

---

## 🗺️ Roadmap (Q3 FY26, because orgs say things like that)

- [x] Rebrand to m.ai (rename repo in Settings → `m.ai`, takes 10 seconds, do it)
- [x] Announce Mythos-class model before building it (this is called "product marketing")
- [x] Mythos engine mode — Kimi K2.6 + persona ✅
- [x] Distillation pipeline ✅
- [ ] Run distillation (needs your Moonshot API key)
- [ ] Train Mythos-potato-350M on Kaggle (weeks of suffering, as is tradition)
- [ ] 1B → 3B → 7B → electricity bill intervention
- [ ] Beat Opus 4.6 — see you in 2045

---

## 🔓 Jailbreaking

Engine mode: take it up with Moonshot, that's their trillion parameters.
Potato mode: there is still nothing to jailbreak. You cannot make a distilled potato evil. It will output slightly better broken Python than before and move on with its life. If you somehow do it — open an issue, we are DYING to know. Hall of Fame (a README comment) awaits. 🥔

---

## 🤝 Contributing

It's one kid pretending to be an org. The org is not hiring. Go away (professionally).

## 📜 License

MIT. Take it. The org has nothing left. The org never had anything.

---

> *"We are thrilled to announce Mythos, our most capable model tier"* — m.ai press release, written in a basement

> *"bro it's just kimi with a system prompt"* — anonymous whistleblower (it was the kid)

> *"Why"* — everyone, still

*Built with 💀 suffering, 🆓 free compute, 🌙 borrowed trillion-parameter wisdom, and absolutely zero corporate structure (the org is a vibe)*

*— m.ai HQ (mom's basement), India 🇮🇳*
