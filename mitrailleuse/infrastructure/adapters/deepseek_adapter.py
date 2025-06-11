"""
DeepSeekAdapter – OpenAI‑compatible wrapper
==========================================
Implements the same send_single / send_batch signature that RequestService
expects.  DeepSeek does **not** expose a batch endpoint (May‑2025), so
send_batch() raises NotImplementedError for now.
"""
from __future__ import annotations
import os
import httpx
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Union

from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit


class DeepSeekAdapter(APIPort):
    BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    ENDPOINT = "/chat/completions"
    TIMEOUT = 60.0  # seconds

    def __init__(self, cfg_or_api_key):
        """
        Accept either a Config object (same as OpenAIAdapter) or raw key str.
        """
        if isinstance(cfg_or_api_key, str):
            self.api_key = cfg_or_api_key
            self.config = {"general": {"proxies": {"proxies_enabled": False}}}
        else:  # config object / dict
            self.api_key = (
                getattr(cfg_or_api_key.deepseek, "api_key", None)
                if hasattr(cfg_or_api_key, "deepseek")
                else cfg_or_api_key.get("deepseek", {}).get("api_key")
            ) or os.getenv("DEEPSEEK_API_KEY", "")
            self.config = cfg_or_api_key
            
        if not self.api_key:
            raise RuntimeError("DeepSeek API key missing (env DEEPSEEK_API_KEY)")
            
        self._log = logging.getLogger(self.__class__.__name__)
        
        # Get proxy configuration from general section
        proxy_config = self.config.get("general", {}).get("proxies", {})
        proxies = None
        if proxy_config.get("proxies_enabled", False):
            proxies = {
                "http://": proxy_config.get("http"),
                "https://": proxy_config.get("https")
            }
        
        # Create async client with proxy configuration
        self._client = httpx.AsyncClient(
            timeout=self.TIMEOUT,
            http2=True,
            transport=httpx.AsyncHTTPTransport(proxy=proxies) if proxies else None
        )

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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @circuit()
    async def ping(self) -> bool:
        """Ping the DeepSeek API to check connectivity."""
        try:
            url = f"{self.BASE_URL}/models"
            r = await self._client.get(url, headers=self._headers())
            return r.status_code == 200
        except Exception as e:
            self._log.error(f"Ping failed: {str(e)}")
            return False

    @circuit()
    async def send_single(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single chat completion request."""
        url = f"{self.BASE_URL}{self.ENDPOINT}"
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
        """DeepSeek does not support batch operations."""
        raise NotImplementedError("DeepSeek has no public batch endpoint yet.")

    async def send_file_batch(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """DeepSeek does not support file batch operations."""
        raise NotImplementedError("DeepSeek has no public file batch endpoint yet.")

    async def get_batch_status(self, job_id: str) -> dict:
        """DeepSeek does not support batch operations."""
        raise NotImplementedError("DeepSeek has no public batch endpoint yet.")

    async def download_batch_results(self, job_id: str, output_dir: Path, task_name: str) -> Path:
        """DeepSeek does not support batch operations."""
        raise NotImplementedError("DeepSeek has no public batch endpoint yet.")
