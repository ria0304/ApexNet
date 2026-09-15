"""
Centralized logging for ApexNet.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")

Behavior is controlled by two env vars so the same code behaves sensibly in
dev and in a deployed/serving context:
    APEXNET_LOG_LEVEL   default "INFO" (DEBUG/INFO/WARNING/ERROR)
    APEXNET_LOG_DIR     default "<project_root>/logs"

Every process (train.py, main.py) gets:
  - a console handler (human-readable, INFO+ by default)
  - a rotating file handler (10MB x 5 backups) capturing everything at
    APEXNET_LOG_LEVEL, so training runs and server sessions are inspectable
    after the fact without re-running anything.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, "logs")


def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("APEXNET_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = os.environ.get("APEXNET_LOG_DIR", DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger("apexnet")
    root.setLevel(level)
    root.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "apexnet.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True
    root.info("logging configured (level=%s, dir=%s)", level_name, log_dir)


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    if not name.startswith("apexnet"):
        name = f"apexnet.{name}"
    return logging.getLogger(name)
