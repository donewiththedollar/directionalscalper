"""Canonical paths for the DS Bridge TUI (derived from package location)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
CONFIGS_DIR = REPO_ROOT / "configs"
