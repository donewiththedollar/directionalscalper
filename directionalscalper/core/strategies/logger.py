import os
import logging
import logging.handlers as handlers
from pathlib import Path

def is_dumb_terminal():
    _term = os.environ.get("TERM", "")
    is_dumb = _term.lower() in ("", "dumb", "unknown")
    return is_dumb

def Logger(
    logger_name: str,
    filename: str,
    level: str = "info",
    console_level: str = "error",  # Changed default value to "error"
    backups: int = 5,
    bytes: int = 5000000,
    stream: bool = False,
):
    log = logging.getLogger(logger_name)
    if log.handlers:
        # Logger is already configured, do not add new handlers
        return log

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = Path("logs")
    file_path.mkdir(exist_ok=True)  # Ensure the directory exists
    log_file = file_path / filename

    logHandler = handlers.RotatingFileHandler(
        log_file, maxBytes=bytes, backupCount=backups
    )
    logHandler.setFormatter(formatter)

    level = logging.getLevelName(level.upper())
    log.setLevel(level)
    log.addHandler(logHandler)

    if stream or not is_dumb_terminal():
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        console_level = logging.getLevelName(console_level.upper())
        consoleHandler.setLevel(console_level)  # Set the level for console output
        log.addHandler(consoleHandler)

    log.propagate = False
    return log