import os
from mitrailleuse.infrastructure.logging.logger import get_logger
log = get_logger(__name__)

def pick_num_processes(cfg) -> int:
    if not cfg.general.multiprocessing_enabled:
        return 1
    return min(cfg.general.num_processes or 10, os.cpu_count() or 1)
