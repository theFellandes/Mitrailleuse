from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Iterable


class APIPort(ABC):
    """Abstract base class for API adapters."""

    @abstractmethod
    async def ping(self) -> bool:
        """Ping the API to check connectivity."""
        pass

    @abstractmethod
    async def send_single(self, payload: dict) -> dict:
        """Send a single request to the API."""
        pass

    @abstractmethod
    async def send_batch(self, payloads: Iterable[dict]) -> Dict[str, Any]:
        """Send a batch of requests to the API."""
        pass

    @abstractmethod
    async def get_batch_status(self, job_id: str) -> dict:
        """Get the status of a batch job."""
        pass

    @abstractmethod
    async def download_batch_results(self, job_id: str, output_dir: Path, task_name: str) -> Path:
        """Download the results of a completed batch job."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the API client and release resources."""
        pass
