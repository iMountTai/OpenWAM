#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# OpenWAM Training — torchrun + Accelerate DeepSpeed
#
# DeepSpeed ZeRO stage is set in train.yaml (training.zero_stage: 2),
# or overridden from the CLI:
#   bash scripts/train.sh training.zero_stage=1
#
# ── Single-node (auto-detect GPUs) ──
# Training defaults to ALL visible GPUs (8-GPU launch on a standard node);
# narrow it only via an explicit NPROC_PER_NODE / CUDA_VISIBLE_DEVICES override.
#   bash scripts/train.sh
#   bash scripts/train.sh training.learning_rate=5e-5
#   NPROC_PER_NODE=4 bash scripts/train.sh
#
# ── Multi-node (env vars set by cloud scheduler) ──
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.1 bash scripts/train.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=192.0.2.1 bash scripts/train.sh
#
# ── Speed preset (loaded by train.py before training imports) ──
# Defaults to presets/fast.env; explicit Hydra selection overrides OPENWAM_PRESET.
#   bash scripts/train.sh training.text_encoder_offload=true
#   bash scripts/train.sh training.speed_preset=null
# ───────────────────────���──────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Quiet logging defaults ──
# Force NCCL_DEBUG to WARN to suppress the per-channel/per-rank INFO spam
# (RingP2P, comm init, topology probes) that buries training progress. We
# unconditionally override here because cloud environments commonly export
# NCCL_DEBUG=INFO by default. Opt back in with OPENWAM_VERBOSE_NCCL=1.
if [[ "${OPENWAM_VERBOSE_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

# ── GPU / Node topology ──
# Priority: explicit override > cloud scheduler vars (HOST_GPU_NUM/HOST_NUM) > auto-detect.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if [[ -n "${HOST_GPU_NUM:-}" ]]; then
        NPROC_PER_NODE="${HOST_GPU_NUM}"
    elif command -v nvidia-smi &>/dev/null; then
        NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
    elif command -v hy-smi &>/dev/null; then
        NPROC_PER_NODE="$(hy-smi 2>/dev/null | grep -E '^[0-9]+' | wc -l)"
    elif command -v rocm-smi &>/dev/null; then
        NPROC_PER_NODE="$(rocm-smi -i 2>/dev/null | grep -c 'GPU\[' || true)"
    else
        NPROC_PER_NODE="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
    fi
fi
if [[ "${NPROC_PER_NODE:-0}" -le 0 ]]; then
    NPROC_PER_NODE=1
fi
NNODES="${NNODES:-${HOST_NUM:-${WORLD_SIZE:-1}}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  OpenWAM Training                                   ║"
echo "║  Nodes: ${NNODES}  GPUs/node: ${NPROC_PER_NODE}  Rank: ${NODE_RANK}            ║"
echo "║  Master: ${MASTER_ADDR}:${MASTER_PORT}                   ║"
echo "╚══════════════════════════════════════════════════════╝"

exec torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/train.py \
    "$@"
