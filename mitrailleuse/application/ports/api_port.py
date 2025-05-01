from abc import ABC, abstractmethod
from typing import Iterable, Any

class APIPort(ABC):
    """Outbound port for any external API."""

    @abstractmethod
    def ping(self) -> bool: ...

    @abstractmethod
    def send_single(self, payload: dict) -> dict: ...

    @abstractmethod
    def send_batch(self, payloads: Iterable[dict]) -> Any: ...
