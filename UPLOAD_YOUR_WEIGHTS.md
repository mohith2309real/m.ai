# 📤 Upload your weights (do this on YOUR Mac)

The cloud can't see your Downloads folder — I'm an org, not a burglar.
Run this on your MacBook in a clone of this repo, and the three
`awaiting-upload` models in `models/registry.json` come alive:

```bash
git clone https://github.com/mohith2309/pitti-rokini   # (or m.ai after rename)
cd pitti-rokini
git lfs install

# your files -> their model slots (from models/registry.json):
cp "/Users/mohith/Downloads/ckpt (1).pt" checkpoints/ropus-4.0.pt      # ROPUS 4.0 (you said so)
cp "/Users/mohith/Downloads/ckpt (3).pt" checkpoints/potus-1.0.pt      # POTUS 1.0 FINAL
cp "/Users/mohith/Downloads/ckpt.pt"     checkpoints/alus-omni-core.pt # ALUS OMNI core

git add checkpoints .gitattributes
git commit -m "weights: ropus-4.0, potus-1.0, alus-omni-core"
git push
```

Swap the assignments any way you like — just keep the destination
filenames matching `models/registry.json` (or edit the registry).

Then, anywhere:

```bash
git lfs pull
pip install torch tiktoken --index-url https://download.pytorch.org/whl/cpu
python mai_run.py ropus-4.0 "prove you are the big one"
```

`ropus-1.0` is an alias of `ropus-4.0`, so the FINAL ships the moment
the big one lands. Corporate efficiency.
