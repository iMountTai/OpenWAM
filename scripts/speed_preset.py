"""Load a Hydra speed preset before importing training dependencies."""

import os
import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _read_speed_preset(path: Path) -> dict[str, str]:
    """Read literal KEY=VALUE assignments; never execute shell syntax.

    Blank lines and full-line comments are ignored. Values may have matching
    outer quotes; variable expansion, escapes and inline comments are not
    interpreted. Validate the complete file before changing the environment.
    """
    settings = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "\x00" in raw_line:
            raise ValueError(f"Invalid speed preset {path}:{line_number}: NUL is not allowed")
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"Invalid speed preset {path}:{line_number}: expected KEY=VALUE "
                "with a valid environment variable name"
            )
        if value[:1] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Invalid speed preset {path}:{line_number}: unmatched value quotes")
            value = value[1:-1]
        settings[key] = value
    return settings


def apply_speed_preset(cfg: DictConfig, project_root: Path) -> dict[str, str]:
    """Apply the resolved Hydra preset before importing torch or OpenWAM.

    CLI/YAML selection takes precedence over OPENWAM_PRESET through the config
    interpolation. Null skips loading without clearing manually set switches.
    File values overwrite existing environment values, matching the old shell
    launcher, and paths are relative to the repository rather than Hydra's cwd.
    """
    selected = OmegaConf.select(cfg, "training.speed_preset", default=None)
    is_main_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"
    if selected is None:
        if is_main_rank:
            print("Speed preset: disabled (existing environment settings are retained)", flush=True)
        return {}
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("training.speed_preset must be a preset file path or null")
    path = Path(selected)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    settings = _read_speed_preset(path)
    os.environ.update(settings)
    if is_main_rank:
        print(f"Speed preset: {path}", flush=True)
        print(
            f"Applied {len(settings)} environment settings (preset values override existing values): "
            + ", ".join(settings),
            flush=True,
        )
    return settings
