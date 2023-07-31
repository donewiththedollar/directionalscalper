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
    backups: int = 5,
    bytes: int = 5000000,
    stream: bool = False,
):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = Path("logs", filename)
    file_path.touch(exist_ok=True)

    logHandler = handlers.RotatingFileHandler(
        file_path, maxBytes=bytes, backupCount=backups
    )
    logHandler.setFormatter(formatter)
        
    level = logging.getLevelName(level.upper())
    log.setLevel(level)
    log.addHandler(logHandler)

    # Remove the console handler
    if not stream or not is_dumb_terminal():
        log.propagate = False
        return log
    
    for h in log.handlers:
        if type(h) is logging.StreamHandler:
            return log
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    log.addHandler(consoleHandler)
    return log



# import logging
# import logging.handlers as handlers
# from pathlib import Path


# def Logger(
#     logger_name: str,
#     filename: str,
#     level: str = "info",
#     backups: int = 5,
#     bytes: int = 5000000,
#     stream: bool = False,
# ):
#     log = logging.getLogger(logger_name)  # Create a unique logger
#     formatter = logging.Formatter(
#         fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     file_path = Path("logs", filename)
#     file_path.touch(exist_ok=True)

#     logHandler = handlers.RotatingFileHandler(
#         file_path, maxBytes=bytes, backupCount=backups
#     )
#     logHandler.setFormatter(formatter)
#     level = logging.getLevelName(level.upper())
#     log.setLevel(level)
#     log.addHandler(logHandler)

#     # Remove the console handler
#     if not stream:
#         log.propagate = False

#     return log