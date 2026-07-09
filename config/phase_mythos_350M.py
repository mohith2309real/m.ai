# nanoGPT config — MYTHOS potato mode (350M, the Mythos-class potato)
# Run: torchrun --nproc_per_node=2 train.py config/phase_mythos_350M.py
#
# Sized for 2x T4 (Kaggle free tier). 350M params:
# 12 layers x 16 heads x 1024 dim would be ~200M; for ~350M we go 24 x 16 x 1024.
# T4s are 16GB — batch is small, gradient accumulation does the heavy lifting.
# This will eat the entire 30 hr/week quota. Multiple weeks. As is tradition.

out_dir = "out-mythos-350M"
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True   # Kaggle WILL kill the session; checkpoint like you mean it

wandb_log = False
wandb_project = "m-ai"
wandb_run_name = "mythos-potato-350M"

dataset = "mythos"              # data/mythos/{train,val}.bin from prepare_nanogpt_data.py
gradient_accumulation_steps = 32
batch_size = 4                  # per GPU; T4 16GB says no to more
block_size = 1024

# ~354M params
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.1                   # distill dataset is small; a little regularization
bias = False

learning_rate = 3e-4
max_iters = 60000
lr_decay_iters = 60000
min_lr = 3e-5
weight_decay = 1e-1
beta2 = 0.95
warmup_iters = 1000

# T4 has no bf16; fp16 with grad scaler
dtype = "float16"
compile = False                 # torch.compile on Kaggle T4 is a coin flip; keep it honest
