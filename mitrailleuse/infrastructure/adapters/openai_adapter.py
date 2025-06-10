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
import random
from openai import RateLimitError, APIError

from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit
from mitrailleuse.infrastructure.logging.logger import get_logger

log = get_logger(__name__)

class OpenAIAdapter(APIPort):
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ENDPOINT = "/chat/completions"
    BATCH_ENDPOINT = "/batch"
    TIMEOUT = 60.0  # seconds

    def __init__(self, config: Dict):
        self.config = config
        self.client = AsyncOpenAI(api_key=config["openai"]["api_key"])
        self.model = config["openai"]["api_information"]["model"]
        self.temperature = config["openai"]["api_information"]["setting"]["temperature"]
        self.max_tokens = config["openai"]["api_information"]["setting"]["max_tokens"]
        self._log = logging.getLogger(self.__class__.__name__)
        
        # Get proxy configuration from general section
        proxy_config = config.get("general", {}).get("proxies", {})
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
            proxies=proxies
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

    @circuit()
    async def send_single(self, payload: Dict) -> Dict:
        """Send a single request to OpenAI with retry and backoff."""
        max_retries = 5
        base_delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=payload["messages"],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response
            except (httpx.RequestError, RateLimitError, APIError) as e:
                log.warning(f"OpenAI send_single attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    log.error(f"OpenAI send_single failed after {max_retries} attempts: {e}")
                    raise
                # Exponential backoff with jitter
                delay = base_delay * (2 ** (attempt - 1))
                delay = delay + random.uniform(0, 0.5 * delay)
                await asyncio.sleep(delay)
            except Exception as e:
                log.error(f"Error in send_single: {str(e)}")
                raise

    async def send_batch(self, payloads: List[Dict]) -> List[Dict]:
        """Send multiple requests to OpenAI in parallel."""
        try:
            tasks = []
            for payload in payloads:
                task = self.send_single(payload)
                tasks.append(task)
            responses = await asyncio.gather(*tasks)
            return responses
        except Exception as e:
            log.error(f"Error in send_batch: {str(e)}")
            raise

    async def send_file_batch(self, file_path: Union[str, Path]) -> Dict:
        """Send a batch request using the OpenAI Batch API."""
        try:
            file_path = Path(file_path)
            
            # Read and prepare input data with custom_ids
            with open(file_path, 'r') as f:
                input_data = [json.loads(line) for line in f if line.strip()]
            
            # Create a new JSONL file with custom_ids and method
            temp_file = file_path.parent / f"temp_{file_path.name}"
            with open(temp_file, 'w') as f:
                for i, item in enumerate(input_data):
                    # Create a new request object
                    request = {
                        'custom_id': f"request_{i}",
                        'method': "POST",
                        'url': "/v1/chat/completions"
                    }
                    
                    # Get prompt content using the configured prompt field
                    prompt_key = self.config["openai"]["prompt"]
                    
                    # Handle different input formats
                    if isinstance(item, str):
                        user_content = item
                    elif isinstance(item, dict):
                        if prompt_key in item:
                            user_content = item[prompt_key]
                        elif len(item) == 1:
                            user_content = next(iter(item.values()))
                        else:
                            user_content = str(item)
                    else:
                        user_content = str(item)
                    
                    # Get system content if dynamic
                    is_dynamic = self.config["openai"]["system_instruction"]["is_dynamic"]
                    sys_prompt_key = self.config["openai"]["system_instruction"]["system_prompt"]
                    system_content = item.get(sys_prompt_key, "You are a helpful assistant") if is_dynamic else "You are a helpful assistant"
                    
                    # Add the body
                    request['body'] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                    
                    f.write(json.dumps(request) + '\n')
            
            # Upload input file
            with open(temp_file, 'rb') as f:
                input_file = await self.client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            # Create batch job
            batch_job = await self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window=self.config["openai"]["batch"]["completion_window"]
            )
            
            # Wait for processing
            while True:
                status = await self.client.batches.retrieve(batch_job.id)
                if status.status in ['completed', 'failed', 'expired', 'cancelled']:
                    break
                await asyncio.sleep(self.config["openai"]["batch"]["batch_check_time"])
            
            # Get results
            if status.status == 'completed':
                results = await self.client.batches.retrieve(batch_job.id)
                output_file_id = results.output_file_id
                
                # Download results
                output_file = await self.client.files.content(output_file_id)
                
                # Handle binary response
                try:
                    # First try to decode as text
                    content = output_file.text
                    try:
                        # Try to parse as JSON
                        batch_results = json.loads(content)
                    except json.JSONDecodeError:
                        # If not JSON, wrap in expected format
                        batch_results = {
                            "choices": [
                                {
                                    "message": {
                                        "content": content
                                    }
                                }
                            ]
                        }
                except Exception as e:
                    # If text decoding fails, try to handle as binary
                    try:
                        content = output_file.content.decode('utf-8')
                        try:
                            batch_results = json.loads(content)
                        except json.JSONDecodeError:
                            batch_results = {
                                "choices": [
                                    {
                                        "message": {
                                            "content": content
                                        }
                                    }
                                ]
                            }
                    except Exception as decode_error:
                        log.error(f"Error decoding binary response: {str(decode_error)}")
                        batch_results = {
                            "choices": [
                                {
                                    "message": {
                                        "content": f"Error decoding response: {str(decode_error)}"
                                    }
                                }
                            ]
                        }
            else:
                batch_results = {
                    "choices": [
                        {
                            "message": {
                                "content": f"Batch job failed with status: {status.status}"
                            }
                        }
                    ]
                }
            
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
            
            return batch_results

        except Exception as e:
            log.error(f"Error in send_file_batch: {str(e)}")
            raise

    async def get_batch_status(self, batch_id: str) -> Dict:
        """Get the status of a batch job."""
        try:
            status = await self.client.batches.retrieve(batch_id)
            return {
                "status": status.status,
                "completed_count": status.completed_count,
                "total_count": status.total_count,
                "created_at": status.created_at
            }
        except Exception as e:
            log.error(f"Error getting batch status: {str(e)}")
            raise

    async def download_batch_results(self, batch_id: str, output_dir: Path, task_name: str) -> Path:
        """Download results from a completed batch job."""
        try:
            results = await self.client.batches.retrieve(batch_id)
            if results.status != 'completed':
                raise Exception(f"Batch job not completed. Status: {results.status}")
            
            output_file_id = results.output_file_id
            output_file = await self.client.files.content(output_file_id)
            
            # Save results
            output_path = output_dir / f"{task_name}_batch_results.jsonl"
            with open(output_path, 'wb') as f:
                f.write(output_file.content)
            
            return output_path
        except Exception as e:
            log.error(f"Error downloading batch results: {str(e)}")
            raise
