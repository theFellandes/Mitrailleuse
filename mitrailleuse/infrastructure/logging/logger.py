import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str, cfg: Optional['Config'] = None):
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Try to configure file logging if config is provided
    if cfg is not None:
        try:
            # Handle different config access patterns
            log_to_file = False
            log_file = "mitrailleuse.log"

            # Try object-style access first
            if hasattr(cfg, 'general') and hasattr(cfg.general, 'logs'):
                log_to_file = getattr(cfg.general.logs, 'log_to_file', False)
                log_file = getattr(cfg.general.logs, 'log_file', log_file)
            # Fall back to dict-style access
            elif isinstance(cfg, dict) and 'general' in cfg and 'logs' in cfg['general']:
                log_to_file = cfg['general']['logs'].get('log_to_file', False)
                log_file = cfg['general']['logs'].get('log_file', log_file)

            if log_to_file:
                path = Path(log_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(path)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
        except Exception as e:
            # Don't let logging configuration errors break the application
            print(f"Warning: Could not configure file logging: {str(e)}")

    return logger
