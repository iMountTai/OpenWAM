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
        VAE_CHANNELS_LAST CONSTANT_CACHE TEXT_MASK TEXT_CACHE VIDEO_PREPROCESS
        FA2_PADDING MOT_SPLIT_ATTN POINTWISE_COMPILE ROPE_FP32 ZERO_OVERLAP TF32
        VAE_HIPDNN VAE_CONCAT_CACHE_WEIGHT VAE_CONCAT_SINGLETON_RESTRIDE
    )
    # Validate before changing the environment.
    for group in "$@"; do
        case "$group" in
            baseline|constants|text_mask|text_cache|video_preprocess|vae_layout|fa2_padding|mot_attention|pointwise|rope_fp32|zero_overlap|tf32|vae_hipdnn|vae_weight_cache|vae_restride) ;;
            *) echo "Unknown OpenWAM optimization group: $group" >&2; return 2 ;;
        esac
    done
    for flag in "${flags[@]}"; do
        export "OPENWAM_OPT_${flag}=0"
    done
    export OPENWAM_VAE_FUSED_CONCAT=1
    for group in "$@"; do
        case "$group" in
            baseline) ;;
            constants) export OPENWAM_OPT_CONSTANT_CACHE=1 ;;
            text_mask) export OPENWAM_OPT_TEXT_MASK=1 ;;
            text_cache) export OPENWAM_OPT_TEXT_CACHE=1 ;;
            video_preprocess) export OPENWAM_OPT_VIDEO_PREPROCESS=1 ;;
            vae_layout) export OPENWAM_OPT_VAE_CHANNELS_LAST=1 ;;
            fa2_padding) export OPENWAM_OPT_FA2_PADDING=1 ;;
            mot_attention) export OPENWAM_OPT_MOT_SPLIT_ATTN=1 ;;
            pointwise) export OPENWAM_OPT_POINTWISE_COMPILE=1 ;;
            rope_fp32) export OPENWAM_OPT_ROPE_FP32=1 ;;
            zero_overlap) export OPENWAM_OPT_ZERO_OVERLAP=1 ;;
            tf32) export OPENWAM_OPT_TF32=1 ;;
            vae_hipdnn|vae_weight_cache|vae_restride)
                export OPENWAM_OPT_VAE_CHANNELS_LAST=1 OPENWAM_OPT_VAE_HIPDNN=1
                [[ "$group" != vae_weight_cache ]] || export OPENWAM_OPT_VAE_CONCAT_CACHE_WEIGHT=1
                [[ "$group" != vae_restride ]] || export OPENWAM_OPT_VAE_CONCAT_SINGLETON_RESTRIDE=1
                ;;
        esac
    done
    return 0
}
_openwam_set_optimization_env "${@:-baseline}"
