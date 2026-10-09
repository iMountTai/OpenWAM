"""Validated optional runtime knobs; absent flags preserve caller configuration."""

import os

from openwam.optimizations import enabled


def integer(name, default, minimum=0):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def checkpoint_layers(total):
    if not enabled("OPENWAM_OPT_PARTIAL_CHECKPOINT"):
        return total
    count = integer("OPENWAM_CHECKPOINT_LAYERS", min(16, total))
    if count > total:
        raise ValueError(f"OPENWAM_CHECKPOINT_LAYERS={count} exceeds model layers={total}")
    return count
