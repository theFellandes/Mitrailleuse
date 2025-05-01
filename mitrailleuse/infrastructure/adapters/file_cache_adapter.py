import json
from pathlib import Path
from mitrailleuse.application.ports.cache_port import CachePort


class FileCache(CachePort):
    def __init__(self, base: Path):
        self.base = base
        self._memory = {}
        self.base.mkdir(parents=True, exist_ok=True)
        self._disk = self.base / "cache.json"
        if self._disk.exists():
            self._memory.update(json.loads(self._disk.read_text()))

    def has(self, key: str) -> bool:
        return key in self._memory

    def get(self, key: str):
        return self._memory[key]

    def set(self, key: str, value):
        self._memory[key] = value

    def flush_to_disk(self):
        self._disk.write_text(json.dumps(self._memory, indent=2))
