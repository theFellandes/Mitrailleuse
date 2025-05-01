import json
import httpx
from typing import Iterable, List, Dict, Any
from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit


class OpenAIAdapter(APIPort):
    BASE = "https://api.openai.com/v1"

    def __init__(self, config):
        self._cfg = config.openai
        self._headers = {"Authorization": f"Bearer {self._cfg.api_key}"}

    @circuit()
    def ping(self) -> bool:
        r = httpx.get(f"{self.BASE}/models", headers=self._headers, timeout=5)
        return r.status_code == 200

    @circuit()
    def send_single(self, payload: dict) -> dict:
        r = httpx.post(f"{self.BASE}/chat/completions",
                       headers=self._headers, json=payload, timeout=30)
        return r.json()

    @circuit()
    def send_batch(self, payloads: Iterable[dict]) -> List[Dict[str, Any]]:
        batch_payload = {"requests": list(payloads)}
        r = httpx.post(f"{self.BASE}/batch",
                       headers=self._headers, json=batch_payload, timeout=None)
        job = r.json()
        # persist job id so we can poll later
        with open("batch_job.json", "w") as fp:
            json.dump(job, fp, indent=2)
        return job

    def get_batch_status(self, job_id: str) -> dict:
        r = httpx.get(f"{self.BASE}/batch/{job_id}",
                      headers=self._headers, timeout=10)
        r.raise_for_status()
        return r.json()
