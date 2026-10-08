#!/usr/bin/env bash
# Source this file before the existing torchrun command. Each invocation resets
# optimization switches, then enables only the named groups.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Usage: source scripts/training_optimization_env.sh baseline|<group> [<group> ...]' >&2
    exit 2
fi

_openwam_set_optimization_env() {
    local flag group
    local -a flags=(
        VAE_CHANNELS_LAST CONSTANT_CACHE TEXT_CACHE FA2_PADDING MOT_SPLIT_ATTN
        POINTWISE_COMPILE ROPE_REAL ZERO_OVERLAP SAC_FFN LIGHTOP_NORM
        VAE_POINTWISE_COMPILE SAC_FFN_INPUT GLOBAL_COMPILE
    )
    # Validate before changing the environment.
    for group in "$@"; do
        case "$group" in
            baseline|constants|text_cache|vae_layout|fa2_padding|mot_attention|pointwise|rope_real|zero_overlap|sac_ffn|lightop_norm|vae_pointwise|sac_ffn_input|global_compile) ;;
            *) echo "Unknown OpenWAM optimization group: $group" >&2; return 2 ;;
        esac
    done
    for flag in "${flags[@]}"; do
        export "OPENWAM_OPT_${flag}=0"
    done
    export OPENWAM_GLOBAL_COMPILE_SCOPE=all
    for group in "$@"; do
        case "$group" in
            baseline) ;;
            constants) export OPENWAM_OPT_CONSTANT_CACHE=1 ;;
            text_cache) export OPENWAM_OPT_TEXT_CACHE=1 ;;
            vae_layout) export OPENWAM_OPT_VAE_CHANNELS_LAST=1 ;;
            fa2_padding) export OPENWAM_OPT_FA2_PADDING=1 ;;
            mot_attention) export OPENWAM_OPT_MOT_SPLIT_ATTN=1 ;;
            pointwise) export OPENWAM_OPT_POINTWISE_COMPILE=1 ;;
            rope_real) export OPENWAM_OPT_ROPE_REAL=1 ;;
            sac_ffn) export OPENWAM_OPT_SAC_FFN=1 ;;
            lightop_norm) export OPENWAM_OPT_LIGHTOP_NORM=1 ;;
            vae_pointwise) export OPENWAM_OPT_VAE_POINTWISE_COMPILE=1 ;;
            sac_ffn_input) export OPENWAM_OPT_SAC_FFN=1 OPENWAM_OPT_SAC_FFN_INPUT=1 ;;
            global_compile) export OPENWAM_OPT_GLOBAL_COMPILE=1 ;;
            zero_overlap) export OPENWAM_OPT_ZERO_OVERLAP=1 ;;
        esac
    done
    return 0
}
_openwam_set_optimization_env "${@:-baseline}"
