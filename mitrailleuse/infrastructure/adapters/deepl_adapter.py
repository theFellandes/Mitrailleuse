"""
DeepLAdapter  ·  May 2025
=========================
Implements send_single() so RequestService can call DeepL just like
OpenAI or DeepSeek.  DeepL has no batch endpoint, so send_batch()
raises NotImplementedError (same pattern as DeepSeek).
"""
from __future__ import annotations
import os
import httpx
import logging
from typing import Dict, Any, List, Union
from pathlib import Path

from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit


class DeepLAdapter(APIPort):
    BASE_URL = os.getenv("DEEPL_BASE_URL", "https://api.deepl.com/v2")
    ENDPOINT = "/translate"
    TIMEOUT = 60.0  # seconds

    def __init__(self, deepl_cfg):
        """
        `deepl_cfg` is DeeplConfig (pydantic) or dict with keys:
            api_key, target_lang, text
        """
        if hasattr(deepl_cfg, 'api_key'):
            # Pydantic model
            self.api_key = deepl_cfg.api_key or os.getenv("DEEPL_API_KEY", "")
            self.text_key = deepl_cfg.text
            self.lang_key = deepl_cfg.target_lang
            self.config = deepl_cfg
        else:  # dict
            self.api_key = deepl_cfg.get("api_key") or os.getenv("DEEPL_API_KEY", "")
            self.text_key = deepl_cfg.get("text", "text")
            self.lang_key = deepl_cfg.get("target_lang", "target_lang")
            self.config = deepl_cfg

        if not self.api_key:
            raise RuntimeError("DeepL API key missing (env DEEPL_API_KEY)")

        self._log = logging.getLogger(self.__class__.__name__)
        
        # Configure proxies
        proxies = None
        if hasattr(self.config, 'general'):
            # Pydantic model
            proxy_config = self.config.general.proxies
            if proxy_config.proxies_enabled:
                proxies = {
                    "http://": proxy_config.http,
                    "https://": proxy_config.https
                }
        else:
            # Dictionary
            proxy_config = self.config.get("general", {}).get("proxies", {})
            if proxy_config.get("proxies_enabled", False):
                proxies = {
                    "http://": proxy_config.get("http"),
                    "https://": proxy_config.get("https")
                }
        
        # Create httpx client with proxy configuration
        self._client = httpx.AsyncClient(
            timeout=self.TIMEOUT,
            http2=True,
            transport=httpx.AsyncHTTPTransport(proxy=proxies) if proxies else None
        )

        # Set environment variables for proxy configuration
        if proxies:
            if proxies.get("http://"):
                os.environ["HTTP_PROXY"] = proxies["http://"]
                os.environ["http_proxy"] = proxies["http://"]
            if proxies.get("https://"):
                os.environ["HTTPS_PROXY"] = proxies["https://"]
                os.environ["https_proxy"] = proxies["https://"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the async client."""
        if hasattr(self, '_client'):
            await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json",
        }

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
            "text": [text_val],  # may send up to 50 items
            "target_lang": lang_val.upper(),  # must be uppercase code
        }

    @circuit()
    async def ping(self) -> bool:
        """Ping the DeepL API to check connectivity."""
        try:
            url = f"{self.BASE_URL}/usage"
            r = await self._client.get(url, headers=self._headers())
            return r.status_code == 200
        except Exception as e:
            self._log.error(f"Ping failed: {str(e)}")
            return False

    @circuit()
    async def send_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single translation request."""
        body = self.build_body(payload)
        url = f"{self.BASE_URL}{self.ENDPOINT}"
        self._log.debug("POST %s -> %s", url, body)
        
        try:
            r = await self._client.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            self._log.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self._log.error(f"Request failed: {str(e)}")
            raise

    async def send_batch(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """DeepL does not support batch operations."""
        raise NotImplementedError("DeepL API does not support batch jobs yet.")

    async def send_file_batch(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """DeepL does not support file batch operations."""
        raise NotImplementedError("DeepL API does not support file batch operations yet.")

    async def get_batch_status(self, job_id: str) -> dict:
        """DeepL does not support batch operations."""
        raise NotImplementedError("DeepL API does not support batch jobs yet.")

    async def download_batch_results(self, job_id: str, output_dir: Path, task_name: str) -> Path:
        """DeepL does not support batch operations."""
        raise NotImplementedError("DeepL API does not support batch jobs yet.")
