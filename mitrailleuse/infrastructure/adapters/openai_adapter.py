"""
OpenAIAdapter  ·  May 2025
=========================
Implements send_single() and send_batch() so RequestService can call OpenAI
just like DeepSeek or DeepL.  OpenAI has a batch endpoint, so send_batch()
is fully implemented.
"""
from __future__ import annotations
import os
import httpx
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import asyncio
from openai import AsyncOpenAI

from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit


class OpenAIAdapter(APIPort):
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ENDPOINT = "/chat/completions"
    BATCH_ENDPOINT = "/batch"
    TIMEOUT = 60.0  # seconds

    def __init__(self, config: Dict):
        self.config = config
        self.client = AsyncOpenAI(api_key=config["openai"]["api_key"])
        self.model = config["openai"]["api_information"]["model"]
        self.settings = config["openai"]["api_information"]["setting"]
        self._log = logging.getLogger(self.__class__.__name__)
        
        # Create async client
        self._client = httpx.AsyncClient(
            timeout=self.TIMEOUT,
            http2=True
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
            "Authorization": f"Bearer {self.config['openai']['api_key']}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI response to extract content."""
        try:
            if isinstance(response, dict):
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"content": content}
            return {"content": str(response)}
        except Exception as e:
            self._log.error(f"Error parsing response: {str(e)}")
            return {"content": str(response)}

    @circuit()
    async def ping(self) -> bool:
        """Ping the OpenAI API to check connectivity."""
        try:
            url = f"{self.BASE_URL}/models"
            r = await self._client.get(url, headers=self._headers())
            return r.status_code == 200
        except Exception as e:
            self._log.error(f"Ping failed: {str(e)}")
            return False

    async def send_single(self, payload: Dict) -> Dict:
        """Send a single request to OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=payload["messages"],
            temperature=self.settings["temperature"],
            max_tokens=self.settings["max_tokens"]
        )
        return response

    async def send_batch(self, payloads: List[Dict]) -> Dict:
        """Send a batch of requests to OpenAI."""
        try:
            messages = []
            for payload in payloads:
                messages.append(payload["messages"][-1])  # Get the last message (user message)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.settings["temperature"],
                max_tokens=self.settings["max_tokens"]
            )
            
            # Convert response to dict format
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'dict'):
                return response.dict()
            elif hasattr(response, 'json'):
                return response.json()
            else:
                # Try to convert to dict if possible
                try:
                    return json.loads(str(response))
                except (json.JSONDecodeError, TypeError):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": str(response)
                                }
                            }
                        ]
                    }
        except Exception as e:
            self._log.error(f"Error in batch processing: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def send_file_batch(self, file_path: Union[str, Path]) -> Dict:
        """Send a file batch request to OpenAI."""
        try:
            # Upload file
            with open(file_path, 'rb') as f:
                file = await self.client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            # Wait for processing
            while True:
                status = await self.client.files.retrieve(file.id)
                if status.status == 'processed':
                    break
                await asyncio.sleep(self.config["openai"]["batch"]["batch_check_time"])
            
            # Get results
            results = await self.client.files.content(file.id)
            
            # Handle binary response
            try:
                # First try to decode as text
                content = results.text
                # Then try to parse as JSON
                try:
                    parsed_results = json.loads(content)
                    return {
                        "id": file.id,
                        "status": "completed",
                        "results": parsed_results
                    }
                except json.JSONDecodeError:
                    # If not JSON, return as text
                    return {
                        "id": file.id,
                        "status": "completed",
                        "results": content
                    }
            except Exception as e:
                # If text decoding fails, try to handle as binary
                try:
                    # Try to decode as UTF-8
                    content = results.content.decode('utf-8')
                    try:
                        parsed_results = json.loads(content)
                        return {
                            "id": file.id,
                            "status": "completed",
                            "results": parsed_results
                        }
                    except json.JSONDecodeError:
                        return {
                            "id": file.id,
                            "status": "completed",
                            "results": content
                        }
                except Exception as decode_error:
                    self._log.error(f"Error decoding binary response: {str(decode_error)}")
                    return {
                        "id": file.id,
                        "status": "failed",
                        "error": f"Failed to decode response: {str(decode_error)}"
                    }
        except Exception as e:
            self._log.error(f"Error in file batch processing: {str(e)}")
            return {
                "id": getattr(file, 'id', None),
                "status": "failed",
                "error": str(e)
            }

    async def get_batch_status(self, batch_id: str) -> Dict:
        """Get the status of a batch job."""
        try:
            status = await self.client.files.retrieve(batch_id)
            return {
                "id": batch_id,
                "status": status.status,
                "created_at": status.created_at,
                "completed_at": status.completed_at,
                "error": status.error if hasattr(status, 'error') else None
            }
        except Exception as e:
            self._log.error(f"Error getting batch status: {str(e)}")
            return {
                "id": batch_id,
                "status": "error",
                "error": str(e)
            }

    async def download_batch_results(self, batch_id: str, output_dir: Path, task_name: str) -> Path:
        """Download batch results."""
        try:
            results = await self.client.files.content(batch_id)
            output_file = output_dir / f"{task_name}_batch_results.jsonl"
            
            # Handle binary response
            try:
                # First try to decode as text
                content = results.text
                try:
                    # Try to parse as JSON
                    parsed_results = json.loads(content)
                    with open(output_file, 'w') as f:
                        json.dump(parsed_results, f, indent=2)
                except json.JSONDecodeError:
                    # If not JSON, write as text
                    with open(output_file, 'w') as f:
                        f.write(content)
            except Exception as e:
                # If text decoding fails, try to handle as binary
                try:
                    content = results.content.decode('utf-8')
                    try:
                        parsed_results = json.loads(content)
                        with open(output_file, 'w') as f:
                            json.dump(parsed_results, f, indent=2)
                    except json.JSONDecodeError:
                        with open(output_file, 'w') as f:
                            f.write(content)
                except Exception as decode_error:
                    self._log.error(f"Error decoding binary response: {str(decode_error)}")
                    raise
            
            return output_file
        except Exception as e:
            self._log.error(f"Error downloading batch results: {str(e)}")
            raise
