"""
DeepLAdapter  ·  May 2025
=========================
Implements send_single() so RequestService can call DeepL just like
OpenAI or DeepSeek.  DeepL has no batch endpoint, so send_batch()
raises NotImplementedError (same pattern as DeepSeek).
"""
from __future__ import annotations
import os, httpx, logging
from typing import Dict, Any, List

from mitrailleuse.infrastructure.utils.circuit_breaker import circuit


class DeepLAdapter:
    BASE_URL = os.getenv("DEEPL_BASE_URL", "https://api.deepl.com/v2")
    ENDPOINT = "/translate"
    TIMEOUT = 60.0  # seconds

    def __init__(self, deepl_cfg):
        """
        `deepl_cfg` is DeeplConfig (pydantic) or dict with keys:
            api_key, target_lang, text
        """
        if isinstance(deepl_cfg, dict):
            self.api_key = deepl_cfg.get("api_key") or os.getenv("DEEPL_API_KEY", "")
            self.text_key = deepl_cfg.get("text", "text")
            self.lang_key = deepl_cfg.get("target_lang", "target_lang")
        else:  # pydantic object
            self.api_key = deepl_cfg.api_key or os.getenv("DEEPL_API_KEY", "")
            self.text_key = deepl_cfg.text
            self.lang_key = deepl_cfg.target_lang

        if not self.api_key:
            raise RuntimeError("DeepL API key missing (env DEEPL_API_KEY)")

        self._log = logging.getLogger(self.__class__.__name__)

    # ───────────────────────── helpers ──────────────────────────
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json",
        }

    # ───────────────────────── single request ───────────────────
    def build_body(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Map fields from payload → DeepL request body."""
        text_val = payload.get(self.text_key, "")
        lang_val = payload.get(self.lang_key, "")
        if not text_val or not lang_val:
            raise ValueError(
                f"Missing translation text or target_lang "
                f"(keys: {self.text_key}, {self.lang_key})"
            )
        return {
            "text": [text_val],  # may send up to 50 items:contentReference[oaicite:1]{index=1}
            "target_lang": lang_val.upper(),  # must be uppercase code:contentReference[oaicite:2]{index=2}
        }

    @circuit()
    def send_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = self.build_body(payload)
        url = f"{self.BASE_URL}{self.ENDPOINT}"
        self._log.debug("POST %s -> %s", url, body)
        with httpx.Client(timeout=self.TIMEOUT, http2=True) as client:
            r = client.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    # ───────────────────────── batch stub ───────────────────────
    def send_batch(self, *_):
        raise NotImplementedError("DeepL API does not support batch jobs yet.")
