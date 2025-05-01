import json
from pathlib import Path
from mitrailleuse.application.ports.cache_port import CachePort
from mitrailleuse.infrastructure.logging.logger import get_logger

log = get_logger(__name__)


class FileCache(CachePort):
    def __init__(self, base: Path):
        self.base = base
        self._memory = {}

        # Ensure cache directory exists
        self.base.mkdir(parents=True, exist_ok=True)

        # Load existing cache if available
        self._disk = self.base / "cache.json"
        if self._disk.exists():
            try:
                with self._disk.open("r") as f:
                    self._memory.update(json.load(f))
                log.info(f"Loaded {len(self._memory)} cache entries from {self._disk}")
            except Exception as e:
                log.error(f"Failed to load cache from {self._disk}: {str(e)}")

    def has(self, key: str) -> bool:
        return key in self._memory

    def get(self, key: str):
        if key not in self._memory:
            raise KeyError(f"Cache key '{key}' not found")
        return self._memory[key]

    def set(self, key: str, value):
        self._memory[key] = value
        log.debug(f"Cached response for {key}")

    def flush_to_disk(self):
        try:
            with self._disk.open("w") as f:
                json.dump(self._memory, f, indent=2)
            log.info(f"Flushed {len(self._memory)} cache entries to {self._disk}")
        except Exception as e:
            log.error(f"Failed to flush cache to disk: {str(e)}")
