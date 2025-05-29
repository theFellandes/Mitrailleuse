import os
from mitrailleuse.infrastructure.logging.logger import get_logger

log = get_logger(__name__)


def pick_num_processes(cfg) -> int:
    """
    Determine the optimal number of processes to use based on config and system resources.
    Supports both object-style and dict-style config access.
    """
    try:
        # Try object-style access
        if hasattr(cfg, "general"):
            general = cfg.general
            multiprocessing_enabled = getattr(general, "multiprocessing_enabled", True)
            if not multiprocessing_enabled:
                return 1
            cpu_count = os.cpu_count() or 1
            cap_percentage = getattr(general, "process_cap_percentage", 75)
            base_cap = max(1, int(cpu_count * (cap_percentage / 100)))
            config_cap = getattr(general, "num_processes", None)
            if config_cap is not None:
                return min(config_cap, base_cap)
            return base_cap
        # Try dict-style access
        general = cfg.get("general", {})
        if not general.get("multiprocessing_enabled", True):
            return 1
        cpu_count = os.cpu_count() or 1
        cap_percentage = general.get("process_cap_percentage", 75)
        base_cap = max(1, int(cpu_count * (cap_percentage / 100)))
        config_cap = general.get("num_processes", None)
        if config_cap is not None:
            return min(config_cap, base_cap)
        return base_cap
    except Exception:
        # Fallback: 75% of CPUs, at least 1
        return max(1, int((os.cpu_count() or 1) * 0.75))
