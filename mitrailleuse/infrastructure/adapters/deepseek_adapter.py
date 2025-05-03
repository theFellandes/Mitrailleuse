"""
DeepSeekAdapter – OpenAI‑compatible wrapper
==========================================
Implements the same send_single / send_batch signature that RequestService
expects.  DeepSeek does **not** expose a batch endpoint (May‑2025), so
send_batch() raises NotImplementedError for now.
"""
from __future__ import annotations
import os, httpx, logging, json, pathlib
from typing import List, Dict, Any

from mitrailleuse.infrastructure.utils.circuit_breaker import circuit


class DeepSeekAdapter:
    BASE_URL   = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    ENDPOINT   = "/chat/completions"
    TIMEOUT    = 60.0        # seconds

    def __init__(self, cfg_or_api_key):
        """
        Accept either a Config object (same as OpenAIAdapter) or raw key str.
        """
        if isinstance(cfg_or_api_key, str):
            self.api_key = cfg_or_api_key
        else:  # config object / dict
            self.api_key = (
                getattr(cfg_or_api_key.deepseek, "api_key", None)
                if hasattr(cfg_or_api_key, "deepseek")
                else cfg_or_api_key.get("deepseek", {}).get("api_key")
            ) or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DeepSeek API key missing (env DEEPSEEK_API_KEY)")
        self._log = logging.getLogger(self.__class__.__name__)

    # ───────────────────────── helpers ──────────────────────────
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    # ───────────────────────── chat completions ─────────────────
    @circuit()
    def send_single(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{self.ENDPOINT}"
        with httpx.Client(timeout=self.TIMEOUT, http2=True) as client:
            r = client.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    # ───────────────────────── batch (not yet) ──────────────────
    def send_batch(self, *_):
        raise NotImplementedError("DeepSeek has no public batch endpoint yet.")

    # downstream code queries for these:
    def get_batch_status(self, *_):          # pragma: no cover
        raise NotImplementedError

    def download_batch_results(self, *_):    # pragma: no cover
        raise NotImplementedError
