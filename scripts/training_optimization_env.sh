#!/usr/bin/env bash
# Source this file before training. Each invocation resets optimization settings,
# then enables only the named groups or the 'hcu_fast' full preset.
# For example:
#   source scripts/training_optimization_env.sh hcu_fast
#   source scripts/training_optimization_env.sh constants linear_bias2d sac_ffn_full=2

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Usage: source scripts/training_optimization_env.sh baseline|hcu_fast|<group> [<group> ...]' >&2
    exit 2
fi

_openwam_set_optimization_env() {
    local flag group
    local -a flags=(
        CONSTANT_CACHE FA2_PADDING FA2_MASK_FUSED PADDING_PLAN_CPU
        LIGHTOP_NORM LINEAR_BIAS2D FUSED_ADAMW ZERO_OVERLAP ZERO_REDUCE_SCATTER
        ZERO_SCALED_PACK ZERO_NORM_OVERLAP ONLINE_ENCODE HOST_ASYNC DEFERRED_GATHER
        HW_QUEUES SAC_FA SAC_FFN_FULL SAC_GEMM
    )

    # Validate before changing the environment.
    for group in "$@"; do
        case "$group" in
            baseline|hcu_fast|constants|fa2_padding|fa2_mask_fused|padding_plan_cpu|lightop_norm|linear_bias2d|fused_adamw|zero_overlap|zero_reduce_scatter|zero_scaled_pack|zero_norm_overlap|online_encode|host_async|deferred_gather|hw_queues|sac_fa|sac_ffn_full|sac_gemm) ;;
            sac_ffn_full=*)
                if [[ ! "${group#*=}" =~ ^[0-9]+$ ]]; then
                    echo "Layer count must be a non-negative integer: $group" >&2; return 2
                fi ;;
            *) echo "Unknown OpenWAM optimization group: $group" >&2; return 2 ;;
        esac
    done

    for flag in "${flags[@]}"; do
        export "OPENWAM_OPT_${flag}=0"
    done
    unset OPENWAM_SAC_LAYERS OPENWAM_SAC_EXTRA_OPS OPENWAM_CHECKPOINT_STATS

    for group in "$@"; do
        case "$group" in
            baseline) ;;
            hcu_fast)
                # Official PR #58 speedups
                export OPENWAM_TEXT_CACHE=1
                export OPENWAM_FAST_INPUT=1
                export OPENWAM_FUSED_ROPE=1
                export OPENWAM_MIXED_ATTN_CKPT=0
                export OPENWAM_VAE_FULLCLIP=1
                export OPENWAM_COMPILE_VAE=1
                export OPENWAM_CONTEXT_TRUNC=1
                export OPENWAM_COMPILE_BLOCKS=1
                export OPENWAM_DS_TIMER_NOSYNC=1
                export OPENWAM_TEXT_MISS_FAST=1
                export OPENWAM_DS_OVERLAP_COMM=1
                export OPENWAM_FAST_CAT=1
                export OPENWAM_GC_FREEZE=1
                export OPENWAM_VAE_CL=1
                export OPENWAM_VAE_FIRST=1
                export OPENWAM_NUMA_BIND=1

                # Hygon DCU / HCU specific optimizations
                export OPENWAM_OPT_CONSTANT_CACHE=1
                export OPENWAM_OPT_FA2_PADDING=1
                export OPENWAM_OPT_FA2_MASK_FUSED=1
                export OPENWAM_OPT_PADDING_PLAN_CPU=1
                export OPENWAM_OPT_LIGHTOP_NORM=1
                export OPENWAM_OPT_LINEAR_BIAS2D=1
                export OPENWAM_OPT_FUSED_ADAMW=1
                export OPENWAM_OPT_ZERO_OVERLAP=1
                export OPENWAM_OPT_ZERO_REDUCE_SCATTER=1
                export OPENWAM_OPT_ZERO_SCALED_PACK=1
                export OPENWAM_OPT_ZERO_NORM_OVERLAP=1
                export OPENWAM_OPT_ONLINE_ENCODE=1
                export OPENWAM_OPT_HOST_ASYNC=1
                export OPENWAM_OPT_DEFERRED_GATHER=1
                export OPENWAM_OPT_HW_QUEUES=1
                ;;
            constants) export OPENWAM_OPT_CONSTANT_CACHE=1 ;;
            fa2_padding) export OPENWAM_OPT_FA2_PADDING=1 ;;
            fa2_mask_fused) export OPENWAM_OPT_FA2_MASK_FUSED=1 ;;
            padding_plan_cpu) export OPENWAM_OPT_PADDING_PLAN_CPU=1 ;;
            lightop_norm) export OPENWAM_OPT_LIGHTOP_NORM=1 ;;
            linear_bias2d) export OPENWAM_OPT_LINEAR_BIAS2D=1 ;;
            fused_adamw) export OPENWAM_OPT_FUSED_ADAMW=1 ;;
            zero_overlap) export OPENWAM_OPT_ZERO_OVERLAP=1 ;;
            zero_reduce_scatter) export OPENWAM_OPT_ZERO_REDUCE_SCATTER=1 ;;
            zero_scaled_pack) export OPENWAM_OPT_ZERO_SCALED_PACK=1 ;;
            zero_norm_overlap) export OPENWAM_OPT_ZERO_NORM_OVERLAP=1 ;;
            online_encode) export OPENWAM_OPT_ONLINE_ENCODE=1 ;;
            host_async) export OPENWAM_OPT_HOST_ASYNC=1 ;;
            deferred_gather) export OPENWAM_OPT_DEFERRED_GATHER=1 ;;
            hw_queues) export OPENWAM_OPT_HW_QUEUES=1 ;;
            sac_fa) export OPENWAM_OPT_SAC_FA=1 ;;
            sac_ffn_full) export OPENWAM_OPT_SAC_FFN_FULL=1 ;;
            sac_ffn_full=*) export OPENWAM_OPT_SAC_FFN_FULL=1 OPENWAM_SAC_LAYERS="${group#*=}" ;;
            sac_gemm) export OPENWAM_OPT_SAC_GEMM=1 ;;
        esac
    done
    return 0
}
_openwam_set_optimization_env "${@:-baseline}"
