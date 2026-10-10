#!/usr/bin/env bash
# Source this file before the existing torchrun command. Each invocation resets
# optimization settings, then enables only the named groups. Two groups take a
# layer count: partial_checkpoint[=M] removes checkpointing from the trailing M
# layers (default 16); sac_ffn_full[=N] keeps complete FFNs for the trailing N
# checkpointed layers (default: all of them). For example:
#   source scripts/training_optimization_env.sh ... sac_ffn_full partial_checkpoint=2
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Usage: source scripts/training_optimization_env.sh baseline|<group> [<group> ...]' >&2
    exit 2
fi

_openwam_set_optimization_env() {
    local flag group
    local -a flags=(
        VAE_CHANNELS_LAST CONSTANT_CACHE TEXT_CACHE FA2_PADDING MOT_SPLIT_ATTN
        POINTWISE_COMPILE ROPE_REAL ZERO_OVERLAP LIGHTOP_NORM
        VAE_POINTWISE_COMPILE GLOBAL_COMPILE
        PARTIAL_CHECKPOINT UINT8_PREPROCESS T5_PREFIX TEXT_TRIM LINEAR_BIAS2D
        SAC_FA SAC_FFN_FULL SAC_GEMM VAE_LAYOUT_REPAIR ZERO_REDUCE_SCATTER ONLINE_ENCODE
    )
    # Validate before changing the environment.
    for group in "$@"; do
        case "$group" in
            baseline|constants|text_cache|vae_layout|fa2_padding|mot_attention|pointwise|rope_real|zero_overlap|lightop_norm|vae_pointwise|global_compile|partial_checkpoint|uint8_preprocess|t5_prefix|text_trim|linear_bias2d|sac_fa|sac_ffn_full|sac_gemm|vae_layout_repair|zero_reduce_scatter|online_encode) ;;
            partial_checkpoint=*|sac_ffn_full=*)
                if [[ ! "${group#*=}" =~ ^[0-9]+$ ]]; then
                    echo "Layer count must be a non-negative integer: $group" >&2; return 2
                fi ;;
            sac_ffn|sac_ffn_input)
                echo "$group was removed; use sac_ffn_full[=N]" >&2; return 2 ;;
            *) echo "Unknown OpenWAM optimization group: $group" >&2; return 2 ;;
        esac
    done
    for flag in "${flags[@]}"; do
        export "OPENWAM_OPT_${flag}=0"
    done
    export OPENWAM_GLOBAL_COMPILE_SCOPE=all
    unset OPENWAM_SAC_LAYERS OPENWAM_SAC_EXTRA_OPS OPENWAM_CHECKPOINT_STATS
    unset OPENWAM_OPT_SAC_FFN OPENWAM_OPT_SAC_FFN_INPUT
    for group in "$@"; do
        case "$group" in
            baseline) ;;
            constants) export OPENWAM_OPT_CONSTANT_CACHE=1 ;;
            text_cache) export OPENWAM_OPT_TEXT_CACHE=1 ;;
            vae_layout) export OPENWAM_OPT_VAE_CHANNELS_LAST=1 ;;
            vae_layout_repair) export OPENWAM_OPT_VAE_LAYOUT_REPAIR=1 ;;
            zero_reduce_scatter) export OPENWAM_OPT_ZERO_REDUCE_SCATTER=1 ;;
            online_encode) export OPENWAM_OPT_ONLINE_ENCODE=1 ;;
            fa2_padding) export OPENWAM_OPT_FA2_PADDING=1 ;;
            mot_attention) export OPENWAM_OPT_MOT_SPLIT_ATTN=1 ;;
            pointwise) export OPENWAM_OPT_POINTWISE_COMPILE=1 ;;
            rope_real) export OPENWAM_OPT_ROPE_REAL=1 ;;
            lightop_norm) export OPENWAM_OPT_LIGHTOP_NORM=1 ;;
            vae_pointwise) export OPENWAM_OPT_VAE_POINTWISE_COMPILE=1 ;;
            global_compile) export OPENWAM_OPT_GLOBAL_COMPILE=1 ;;
            zero_overlap) export OPENWAM_OPT_ZERO_OVERLAP=1 ;;
            partial_checkpoint) export OPENWAM_OPT_PARTIAL_CHECKPOINT=16 ;;
            partial_checkpoint=*) export OPENWAM_OPT_PARTIAL_CHECKPOINT="${group#*=}" ;;
            uint8_preprocess) export OPENWAM_OPT_UINT8_PREPROCESS=1 ;;
            t5_prefix) export OPENWAM_OPT_T5_PREFIX=1 ;;
            text_trim) export OPENWAM_OPT_TEXT_TRIM=1 ;;
            linear_bias2d) export OPENWAM_OPT_LINEAR_BIAS2D=1 ;;
            sac_fa) export OPENWAM_OPT_SAC_FA=1 ;;
            sac_ffn_full) export OPENWAM_OPT_SAC_FFN_FULL=1 ;;
            sac_ffn_full=*) export OPENWAM_OPT_SAC_FFN_FULL=1 OPENWAM_SAC_LAYERS="${group#*=}" ;;
            sac_gemm) export OPENWAM_OPT_SAC_GEMM=1 ;;
        esac
    done
    return 0
}
_openwam_set_optimization_env "${@:-baseline}"
