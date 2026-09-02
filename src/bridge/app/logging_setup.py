"""Structured logging: console + rotating file."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: Path | None = Path("logs")) -> logging.Logger:
    root = logging.getLogger("bridge")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if root.handlers:
        return root
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                Path(log_dir) / "bridge.log", maxBytes=2_000_000, backupCount=3
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            root.warning("Could not open log directory %s", log_dir)
    return root
