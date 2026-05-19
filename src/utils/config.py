from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data: Optional[Dict[str, Any]] = yaml.safe_load(f)
    return Config(raw=data or {})
