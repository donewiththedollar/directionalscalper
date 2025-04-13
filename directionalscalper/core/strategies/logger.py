# logger.py
import os
import logging
import logging.handlers as handlers
from pathlib import Path

def is_dumb_terminal():
    _term = os.environ.get("TERM", "")
    return _term.lower() in ("", "dumb", "unknown")

def Logger(
    logger_name: str,
    filename: str,
    level: str = "info",
    console_level: str = "error",
    backups: int = 5,
    bytes: int = 5000000,
    stream: bool = False,
):
    """
    Create or return a logger that writes to a rotating file, plus optional console logs.
    Includes millisecond timestamps in file/console output.
    """

    # If the logger already exists with handlers, return it to avoid adding duplicate handlers.
    log = logging.getLogger(logger_name)
    if log.handlers:
        return log

    #
    # OPTIONAL: Change standard level names to single-letter. You can remove if you prefer "INFO", "ERROR", etc.
    #
    # logging.addLevelName(logging.INFO, "I")
    # logging.addLevelName(logging.ERROR, "E")
    # logging.addLevelName(logging.WARNING, "W")
    # logging.addLevelName(logging.DEBUG, "D")
    # logging.addLevelName(logging.CRITICAL, "C")
    #

    #
    # 1) Create a formatter with millisecond timestamps
    #
    # Example line: "2025-03-25 13:50:27.123 - BybitBaseStrategy - INFO - Some message"
    #
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    #
    # 2) Create a rotating file handler (by size)
    #
    file_path = Path("logs")
    file_path.mkdir(exist_ok=True)  # Ensure the logs/ directory exists
    log_file = file_path / filename

    logHandler = handlers.RotatingFileHandler(
        log_file, maxBytes=bytes, backupCount=backups
    )
    logHandler.setFormatter(formatter)

    #
    # 3) Set the logger’s file level
    #
    level = logging.getLevelName(level.upper())
    log.setLevel(level)
    log.addHandler(logHandler)

    #
    # 4) Optional console stream
    #
    if stream or not is_dumb_terminal():
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        console_level = logging.getLevelName(console_level.upper())
        consoleHandler.setLevel(console_level)
        log.addHandler(consoleHandler)

    log.propagate = False
    return log
