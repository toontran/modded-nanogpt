#!/usr/bin/env bash
set -euo pipefail

NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-records/track_3_small_optimization/train_gpt_modded.py}"
LOGDIR="${LOGDIR:-sweep_logs}"
SEED="${SEED:-1234}"
TRAIN_STEPS="${TRAIN_STEPS:-3800}"

# Manually chosen constant: total number of machines sharing the sweep
NUM_MACHINES="${NUM_MACHINES:-4}"

mkdir -p "$LOGDIR"

usage() {
    local total_configs
    total_configs=$(count_all_configs)

    cat <<EOF
Usage: $0 <machine_index>

machine_index should be between 0 and $((NUM_MACHINES - 1))

Current settings:
  NUM_MACHINES = $NUM_MACHINES
  NGPU         = $NGPU
  TRAIN_SCRIPT = $TRAIN_SCRIPT
  SEED         = $SEED
  TRAIN_STEPS  = $TRAIN_STEPS

Total sweep configurations: $total_configs

Work partitioning:
  Each configuration is assigned a global experiment number in:
    [0, $((total_configs - 1))]

  A configuration runs on this machine iff:
    experiment_number % NUM_MACHINES == machine_index

This gives near-equal partitioning across machines.
EOF
}

run() {
    local name="$1"
    shift

    echo "=== $name ==="
    # echo "torchrun --standalone --nproc_per_node=\"$NGPU\" \"$TRAIN_SCRIPT\" --seed \"$SEED\" --train_steps \"$TRAIN_STEPS\" $*"

    torchrun --standalone --nproc_per_node="$NGPU" "$TRAIN_SCRIPT" \
        --seed "$SEED" \
        --train_steps "$TRAIN_STEPS" \
        "$@" 2>&1 | tee "$LOGDIR/$name.txt"
}

muon_grid() {
python3 <<'PY'
import itertools

lr = [0.01, 0.02, 0.03]
wd = [0.003, 0.01, 0.03]
mom = [0.95]

for l, w, m in itertools.product(lr, wd, mom):
    print(f"muon --hidden_lr {l} --hidden_weight_decay {w} --hidden_momentum {m} --hidden_nesterov true")
PY
}

adamw_grid() {
python3 <<'PY'
import itertools

lr = [0.0005, 0.001, 0.0015, 0.002]
wd = [0.03, 0.125, 0.25]
beta2 = [0.95, 0.99]
eps = [1e-8, 1e-10]

for l, w, b2, e in itertools.product(lr, wd, beta2, eps):
    print(f"adamw --hidden_lr {l} --hidden_weight_decay {w} --hidden_beta1 0.9 --hidden_beta2 {b2} --hidden_eps {e} --hidden_warmup_tokens 250")
PY
}

# pkron_grid() {
# python3 <<'PY'
# import itertools
#
# lr = [0.0002, 0.0005, 0.001]
# wd = [0.1, 0.625, 1.0]
#
# for l, w in itertools.product(lr, wd):
#     print(f"pkron --hidden_lr {l} --hidden_weight_decay {w}")
# PY
# }

all_configs() {
    muon_grid
    adamw_grid
    # pkron_grid
}

count_all_configs() {
    all_configs | wc -l | tr -d ' '
}

# ----------------------------
# Argument handling
# ----------------------------
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

machine_index="$1"

if ! [[ "$machine_index" =~ ^[0-9]+$ ]]; then
    echo "Error: machine_index must be a nonnegative integer."
    usage
    exit 1
fi

if [ "$machine_index" -ge "$NUM_MACHINES" ]; then
    echo "Error: machine_index must be between 0 and $((NUM_MACHINES - 1))."
    usage
    exit 1
fi

total_configs=$(count_all_configs)
if [ "$total_configs" -eq 0 ]; then
    echo "Error: no configurations generated."
    exit 1
fi

echo "Machine index: $machine_index / $((NUM_MACHINES - 1))"
echo "NUM_MACHINES: $NUM_MACHINES"
echo "Total configs: $total_configs"
echo

# ----------------------------
# Sweep only this machine's share
# ----------------------------
global_idx=0
assigned_count=0

while IFS= read -r line; do
    [ -z "$line" ] && continue

    if [ $((global_idx % NUM_MACHINES)) -ne "$machine_index" ]; then
        global_idx=$((global_idx + 1))
        continue
    fi

    opt="${line%% *}"
    config="${line#* }"

    name="${opt}_exp${global_idx}_$(echo "$config" | tr ' ' '_' | tr '=' '-' | tr '.' 'p')"

    echo "Running assigned config:"
    echo "  global_idx = $global_idx"
    echo "  optimizer  = $opt"
    echo "  args       = $config"
    echo

    run "$name" --hidden_opt "$opt" $config

    assigned_count=$((assigned_count + 1))
    global_idx=$((global_idx + 1))
done < <(all_configs)

echo
echo "Done."
echo "Machine $machine_index ran $assigned_count configuration(s)."