#!/usr/bin/env bash
set -euo pipefail

NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-records/track_3_small_optimization/train_gpt_modded.py}"
LOGDIR="${LOGDIR:-sweep_logs}"
mkdir -p "$LOGDIR"

SEED="${SEED:-1234}"
TRAIN_STEPS="${TRAIN_STEPS:-3800}"

run() {
    local name="$1"
    shift
    echo "=== $name ==="
    echo "torchrun --standalone --nproc_per_node=\"$NGPU\" \"$TRAIN_SCRIPT\" --seed \"$SEED\" --train_steps \"$TRAIN_STEPS\" \"$@\" 2>&1 | tee \"$LOGDIR/$name.txt\""
    # torchrun --standalone --nproc_per_node="$NGPU" "$TRAIN_SCRIPT" \
    #     --seed "$SEED" \
    #     --train_steps "$TRAIN_STEPS" \
    #     "$@" 2>&1 | tee "$LOGDIR/$name.txt"
}

grid_run() {
    local opt="$1"
    shift

    while IFS= read -r config; do
        local name
        name="${opt}_$(echo "$config" | tr ' ' '_' | tr '=' '-')"
        run "$name" --hidden_opt "$opt" $config
    done <<< "$1"
}

# --------------------------
# Muon grid
# --------------------------

muon_grid() {
python << 'PY'
import itertools

lr = [0.01,0.02,0.03]
wd = [0.003,0.01,0.03]
mom = [0.95]

for l,w,m in itertools.product(lr,wd,mom):
    print(f"--hidden_lr {l} --hidden_weight_decay {w} --hidden_momentum {m} --hidden_nesterov true")
PY
}

grid_run muon "$(muon_grid)"

# --------------------------
# AdamW grid
# --------------------------

adamw_grid() {
python << 'PY'
import itertools

lr=[0.0005,0.001,0.0015,0.002]
wd=[0.03,0.125,0.25]
beta2=[0.95,0.99]
eps=[1e-8,1e-10]

for l,w,b2,e in itertools.product(lr,wd,beta2,eps):
    print(f"--hidden_lr {l} --hidden_weight_decay {w} --hidden_beta1 0.9 --hidden_beta2 {b2} --hidden_eps {e}")
PY
}

grid_run adamw "$(adamw_grid)"

# --------------------------
# pKron grid
# --------------------------

# pkron_grid() {
# python << 'PY'
# import itertools

# lr=[0.0002,0.0005,0.001]
# wd=[0.1,0.625,1.0]

# for l,w in itertools.product(lr,wd):
#     print(f"--hidden_lr {l} --hidden_weight_decay {w}")
# PY
# }

# grid_run pkron "$(pkron_grid)"
