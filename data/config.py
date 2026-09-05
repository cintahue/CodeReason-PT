from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return json.loads(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def resolve_phase0_paths(
    config: dict[str, Any],
    *,
    raw_dir: str | None = None,
    artifact_root: str | None = None,
) -> dict[str, Path]:
    configured_paths = config.get("paths", {})
    root = Path(artifact_root or configured_paths["artifact_root"]).expanduser()
    raw = Path(raw_dir or configured_paths["raw_dir"]).expanduser()
    return {
        "raw_dir": raw,
        "artifact_root": root,
        "processed_dir": root / "processed",
        "reports_dir": root / "reports",
        "logs_dir": root / "logs",
    }


def ensure_phase0_dirs(paths: dict[str, Path]) -> None:
    for key in ("artifact_root", "processed_dir", "reports_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

