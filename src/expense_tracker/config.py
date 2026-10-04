"""Central configuration and logging setup."""

import json
import logging
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """Read config.json. Relative paths inside it are resolved from the project root."""
    config_file = Path(path) if path else CONFIG_PATH
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    config["paths"] = {
        key: str(PROJECT_ROOT / value) for key, value in config["paths"].items()
    }
    return config


def setup_logging(level: str | None = None) -> None:
    """Log to the console and to a file (created on first use)."""
    config = load_config()["logging"]
    log_file = PROJECT_ROOT / config["file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, (level or config["level"]).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
