import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any


def get_logger(name: str, cfg: Optional[Union[Dict[str, Any], 'Config']] = None):
    """
    Get a configured logger for the given name.

    Args:
        name: Logger name (usually __name__)
        cfg: Optional configuration object or dictionary

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, it's already configured
    if logger.handlers:
        return logger

    # Always set the logger to INFO level or higher
    logger.setLevel(logging.INFO)

    # Create a standard formatter
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Always add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Try to configure file logging if config is provided
    if cfg is not None:
        try:
            # Initialize defaults
            log_to_file = False
            log_file = "mitrailleuse.log"

            # Try different access patterns based on the type of cfg
            if hasattr(cfg, 'general') and hasattr(cfg.general, 'logs'):
                log_to_file = getattr(cfg.general.logs, 'log_to_file', False)
                log_file = getattr(cfg.general.logs, 'log_file', log_file)
            elif isinstance(cfg, dict) and 'general' in cfg:
                if isinstance(cfg['general'], dict) and 'logs' in cfg['general']:
                    log_to_file = cfg['general']['logs'].get('log_to_file', False)
                    log_file = cfg['general']['logs'].get('log_file', log_file)

            # Add file handler if enabled
            if log_to_file:
                log_path = Path(log_file)

                # Create parent directories if needed
                if log_path.parent and not log_path.parent.exists():
                    log_path.parent.mkdir(parents=True, exist_ok=True)

                # Configure file handler
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setFormatter(fmt)
                logger.addHandler(file_handler)
                logger.info(f"File logging enabled at {log_path}")
        except Exception as e:
            # Don't let logging configuration errors break the application
            print(f"Warning: Could not configure file logging: {str(e)}")

    # If task folder is in the logger name, try to set up task-specific logging
    if "task_folder" in name or "/tasks/" in name:
        try:
            # Try to extract the task path from the logger name
            parts = name.split('task_folder=')
            if len(parts) > 1:
                task_folder = parts[1].split(')', 1)[0].strip("'\"")
                task_log_path = Path(task_folder) / "logs" / "log.log"

                # Create parent directories if needed
                task_log_path.parent.mkdir(parents=True, exist_ok=True)

                # Add task-specific file handler
                task_handler = logging.FileHandler(task_log_path, encoding='utf-8')
                task_handler.setFormatter(fmt)
                logger.addHandler(task_handler)
                logger.info(f"Task-specific logging enabled at {task_log_path}")
        except Exception as e:
            print(f"Warning: Could not configure task-specific logging: {str(e)}")

    return logger


def setup_task_logger(task_path: Path) -> logging.Logger:
    """
    Create a logger specifically for a task, writing to {task_path}/logs/log.log

    Args:
        task_path: Path to the task folder

    Returns:
        Configured logger instance
    """
    logger_name = f"task_{task_path.name}"
    logger = logging.getLogger(logger_name)

    # If already configured, return it
    if logger.handlers:
        return logger

    # Configure the logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler in the task's logs directory
    logs_dir = task_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "log.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Task logger initialized for {task_path.name}")
    return logger
