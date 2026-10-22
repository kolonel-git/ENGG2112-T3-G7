"""Loading of configs/protocol.yaml with a guard against undecided (TODO) values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "configs" / "protocol.yaml"


def load_protocol(path: Path | str = DEFAULT_PROTOCOL) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Return cfg[a][b]... for 'a.b'; raise if missing or still marked TODO."""
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"{dotted_key} missing from protocol")
        value = value[part]
    if value == "TODO" or value is None:
        raise ValueError(f"{dotted_key} is still TODO in configs/protocol.yaml; decide it first")
    return value
