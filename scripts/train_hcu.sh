#!/usr/bin/env bash
# HCU fine-tuning: 8 cards, BF16, ZeRO-2, blocks_ranking_size, 30 steps.
# Usage: bash scripts/train_hcu.sh [training.max_steps=300] [Hydra overrides...]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
resource_root="${OPENWAM_RESOURCE_ROOT:-$repo_root}"
# A clean Git worktree can reuse the main checkout's assets without copying them.
if [[ -z "${OPENWAM_RESOURCE_ROOT:-}" && ! -d "$repo_root/assets" ]]; then
    common_dir="$(git rev-parse --git-common-dir)"
    resource_root="$(cd "$common_dir/.." && pwd)"
fi
cd "$resource_root"

source "$repo_root/scripts/training_optimization_env.sh" \
    vae_layout rope_real fa2_padding mot_attention text_cache zero_overlap \
    constants lightop_norm sac_ffn_input global_compile

# Ignore inherited tuning/profiling settings for the default-library run.
unset ROCBLAS_TENSILE_LIBPATH ROCBLAS_LAYER ROCBLAS_LOG_BENCH_PATH \
    ROCBLAS_LOG_TRACE_PATH ROCBLAS_LOG_PROFILE_PATH
export enable_profiling=0 ENABLE_PROFILING=0 TORCHINDUCTOR_COMPILE_THREADS=1
if [[ "${OPENWAM_VERBOSE_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

exec torchrun \
    --nnodes "${NNODES:-${HOST_NUM:-1}}" \
    --nproc_per_node "${NPROC_PER_NODE:-${HOST_GPU_NUM:-8}}" \
    --node_rank "${NODE_RANK:-${RANK:-0}}" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29500}" \
    "$repo_root/scripts/train.py" \
    dataloader=robotwin \
    dataloader.dataset_dir=assets/benchmark_data/robotwin2.0/dataset \
    dataloader.embodiment=aloha-agilex dataloader.variant=clean_50 \
    'dataloader.tasks=[blocks_ranking_size]' \
    +dataloader.normalization_stats_path=outputs/hcu_alpha_20260930/robotwin_blocks_ranking_size_stats.npy \
    training.finetune_ckpt_path=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.num_epochs=null training.max_steps=30 training.batch_size=16 \
    training.mixed_precision=bf16 training.zero_stage=2 \
    training.dataset_num_workers=8 training.save_steps=100 \
    training.save_full_states_for_resume=false \
    "training.output_path=$repo_root/outputs/hcu_rope_real_compile" \
    project.wandb.project=null project.seed=42 \
    "$@"
