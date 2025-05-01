import json
import httpx
from typing import Iterable, List, Dict, Any
from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit

log = get_logger(__name__)


class OpenAIAdapter(APIPort):
    BASE = "https://api.openai.com/v1"

    def __init__(self, config):
        self._cfg = config.openai
        self._headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json"
        }
        # Configure proxies if enabled
        self._proxy = None
        if getattr(self._cfg.proxies, "proxies_enabled", False):
            self._proxy = (
                getattr(self._cfg.proxies, "https", None)
                or getattr(self._cfg.proxies, "http", None)
            )
            log.info(f"Using proxy: {self._proxy}")

    @circuit()
    def ping(self) -> bool:
        log.info("Pinging OpenAI API")
        try:
            r = httpx.get(
                f"{self.BASE}/models",
                headers=self._headers,
                proxy=self._proxy,
                timeout=5
            )
            return r.status_code == 200
        except Exception as e:
            log.error(f"Ping failed: {str(e)}")
            return False

    @circuit()
    def send_single(self, payload: dict) -> dict:
        log.info(f"Sending single request: {payload.get('model', 'unknown')}")
        try:
            r = httpx.post(
                f"{self.BASE}/chat/completions",
                headers=self._headers,
                json=payload,
                proxy=self._proxy,
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"Request failed: {str(e)}")
            raise

    @circuit()
    def send_batch(self, payloads: Iterable[dict]) -> Dict[str, Any]:
        log.info(f"Sending batch request with {sum(1 for _ in payloads)} items")
        # Ensure payloads is a list, as iterables can only be consumed once
        payloads_list = list(payloads)
        batch_payload = {"requests": payloads_list}

        try:
            r = httpx.post(
                f"{self.BASE}/batch",
                headers=self._headers,
                json=batch_payload,
                proxy=self._proxy,
                timeout=None
            )
            r.raise_for_status()
            job = r.json()
            log.info(f"Batch job created: {job.get('id', 'unknown')}")
            return job
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"Batch request failed: {str(e)}")
            raise

    def get_batch_status(self, job_id: str) -> dict:
        log.info(f"Checking status for batch job: {job_id}")
        try:
            r = httpx.get(
                f"{self.BASE}/batch/{job_id}",
                headers=self._headers,
                proxy=self._proxy,
                timeout=10
            )
            r.raise_for_status()
            status = r.json()
            log.info(f"Batch status: {status.get('status', 'unknown')}")
            return status
        except Exception as e:
            log.error(f"Failed to get batch status: {str(e)}")
            raise

