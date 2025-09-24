"""
Logging utilities 
"""
import logging
import os
from typing import Optional


def setup_logging() -> logging.Logger:
    """Set up logging based on environment variables."""
    logger = logging.getLogger("src")
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Get log level from environment (0=silent, 1=info, 2=debug)
    log_level_env = os.getenv("LOG_LEVEL", "0")
    try:
        log_level_num = int(log_level_env)
    except ValueError:
        log_level_num = 0
    
    # Map to logging levels
    if log_level_num == 0:
        logger.setLevel(logging.CRITICAL + 1)  # Effectively silent
        return logger
    elif log_level_num == 1:
        logger.setLevel(logging.INFO)
    else:  # log_level_num >= 2
        logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up file handler if LOG_FILE is specified
    log_file = os.getenv("LOG_FILE")
    if log_file:
        try:
            # Validate we can write to the file path eagerly
            # This will raise if directories don't exist or permissions are insufficient
            with open(log_file, 'a'):
                pass
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Per specification: invalid log file path must cause startup failure
            raise SystemExit(1) from e
    else:
        # Default to stderr (not stdout to avoid interfering with NDJSON output)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    return logging.getLogger("src")