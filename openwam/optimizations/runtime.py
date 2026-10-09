"""Validated optional runtime knobs; absent flags preserve caller configuration."""

import os


def integer(name, default, minimum=0):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def checkpoint_skip_layers():
    """Number of trailing layers without checkpointing; unset/false means zero."""
    name = "OPENWAM_OPT_PARTIAL_CHECKPOINT"
    if os.environ.get(name, "").strip().lower() in {"", "false", "no", "off"}:
        return 0
    return integer(name, 0)


def checkpoint_layers(total):
    skip = checkpoint_skip_layers()
    if skip > total:
        raise ValueError(f"OPENWAM_OPT_PARTIAL_CHECKPOINT={skip} exceeds model layers={total}")
    return total - skip
