import logging
import os
import sys


def setup_logging() -> logging.Logger:
    # set up logging based on environment variables
    logger = logging.getLogger("src")

    # clear any existing handlers
    logger.handlers.clear()

    # validate LOG_FILE first, regardless of log level
    log_file = os.getenv("LOG_FILE")
    if log_file:
        try:
            # validate we can write to the file path eagerly
            with open(log_file, "r+"):
                pass
        except Exception as e:
            # invalid log file path cause startup failure
            try:
                logger = get_logger()
                logger.critical(f"Error: Invalid LOG_FILE path '{log_file}': {e}")
            except:
                #fallback if logger isn't configured yet
                print(f"Error: Invalid LOG_FILE path '{log_file}': {e}", file=sys.stderr)
            sys.exit(1)

    # get log level from environment (0=silent, 1=info, 2=debug)
    log_level_env = os.getenv("LOG_LEVEL", "0")
    try:
        log_level_num = int(log_level_env)
    except ValueError:
        # use logger if available, otherwise stderr
        try:
            logger = get_logger()
            logger.critical(f"Error: LOG_LEVEL must be an integer, got '{log_level_env}'")
        except:
            print(f"Error: LOG_LEVEL must be an integer, got '{log_level_env}'", file=sys.stderr)
        sys.exit(1)

    # validate LOG_LEVEL is in {0,1,2}
    if log_level_num not in {0, 1, 2}:
        # use logger if available, otherwise stderr
        try:
            logger = get_logger()
            logger.critical(f"Error: LOG_LEVEL must be 0, 1, or 2, got {log_level_num}")
        except:
            print(f"Error: LOG_LEVEL must be 0, 1, or 2, got {log_level_num}", file=sys.stderr)
        sys.exit(1)

    # map to logging levels
    if log_level_num == 0:
        logger.setLevel(logging.CRITICAL + 1)
        return logger
    elif log_level_num == 1:
        logger.setLevel(logging.INFO)
    else:  # log_level_num == 2
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up file handler if LOG_FILE is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # default to stderr
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    # get the configured logger instance
    return logging.getLogger("src")
