"""Validated optional runtime knobs; absent flags preserve caller configuration."""

from openwam.optimizations.config import option, settings


def integer(name, default, minimum=0):
    raw = option(name)
    value = default if raw is None else raw
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"optimizations.{name} must be an integer, got {raw!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def checkpoint_skip_layers():
    """Number of trailing layers without checkpointing; unset/false means zero."""
    return settings().checkpoint_skip_layers


def checkpoint_layers(total):
    skip = checkpoint_skip_layers()
    if skip > total:
        raise ValueError(f"checkpoint_skip_layers={skip} exceeds model layers={total}")
    return total - skip
