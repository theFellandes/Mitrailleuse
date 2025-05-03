from abc import ABC, abstractmethod
from typing import Any


class CachePort(ABC):
    @abstractmethod
    def has(self, key: str) -> bool: ...

    @abstractmethod
    def get(self, key: str) -> Any: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def flush_to_disk(self) -> None: ...
