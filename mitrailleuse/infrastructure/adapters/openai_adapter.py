import json
import time
from pathlib import Path

import httpx
from typing import Iterable, List, Dict, Any
from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.utils.circuit_breaker import circuit

log = get_logger(__name__)


class OpenAIAdapter(APIPort):
    BASE = "https://api.openai.com/v1"
    BATCH_ENDPOINT = f"{BASE}/batches"

    def __init__(self, config):
        self._cfg = config.openai if hasattr(config, 'openai') else config['openai']
        self._headers = {
            "Authorization": f"Bearer {self._cfg.api_key if hasattr(self._cfg, 'api_key') else self._cfg['api_key']}",
            "Content-Type": "application/json"
        }
        # Configure proxies if enabled
        self._proxy = None
        try:
            # Handle both object and dict style config
            proxies_enabled = getattr(self._cfg.proxies, "proxies_enabled", False) if hasattr(self._cfg,
                                                                                              'proxies') else self._cfg.get(
                'proxies', {}).get('proxies_enabled', False)

            if proxies_enabled:
                if hasattr(self._cfg, 'proxies'):
                    self._proxy = getattr(self._cfg.proxies, "https", None) or getattr(self._cfg.proxies, "http", None)
                else:
                    self._proxy = self._cfg.get('proxies', {}).get('https') or self._cfg.get('proxies', {}).get('http')

                log.info(f"Using proxy: {self._proxy}")
        except Exception as e:
            log.warning(f"Error configuring proxies: {str(e)}")

    @staticmethod
    def _build_jsonl_file(payloads: list[dict], temp_dir: Path) -> Path:
        """Create a JSONL file from a list of payload dictionaries.

        For batch processing, each line should be a valid request object
        that can be sent to the chat completions endpoint.
        """
        # Ensure the directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique filename using current timestamp
        timestamp = int(time.time())
        jsonl_path = temp_dir / f"batch_inputs_{timestamp}.jsonl"

        # Write each payload as a JSON line
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, payload in enumerate(payloads):
                request_line = {
                    "custom_id": f"item-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": payload  # includes "model"
                }
                f.write(json.dumps(request_line, ensure_ascii=False) + "\n")

        log.info(f"Created batch input file at {jsonl_path} with {len(payloads)} items")
        return jsonl_path

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
        """Send a batch of requests using OpenAI's batch API."""
        # Convert iterable to list for multiple passes
        payloads_list = list(payloads)
        log.info(f"Sending batch request with {len(payloads_list)} items")

        try:
            # Create a temporary directory for the JSONL file if needed
            temp_dir = Path("./temp_batch_files")
            temp_dir.mkdir(exist_ok=True)

            # 1. Create JSONL file
            jsonl_path = self._build_jsonl_file(payloads_list, temp_dir)

            # 2. Upload file to OpenAI
            with open(jsonl_path, "rb") as f:
                files = {"file": (jsonl_path.name, f, "application/jsonl")}
                upload_response = httpx.post(
                    f"{self.BASE}/files",
                    headers={"Authorization": self._headers["Authorization"]},
                    files=files,
                    data={"purpose": "batch"}
                )
                upload_response.raise_for_status()
                file_id = upload_response.json()["id"]
                log.info(f"Uploaded batch file with ID: {file_id}")

            # 3. Create batch job with correct parameters based on error feedback
            # Get model from the first payload if available
            model = None
            if payloads_list and "model" in payloads_list[0]:
                model = payloads_list[0]["model"]

            # Create the batch request payload with correct parameter names
            batch_data = {
                "input_file_id": file_id,  # Use input_file_id as requested by the API
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions"
            }

            batch_response = httpx.post(
                self.BATCH_ENDPOINT,
                headers=self._headers,
                json=batch_data,
                timeout=60
            )
            batch_response.raise_for_status()
            batch_job = batch_response.json()

            log.info(f"Created batch job with ID: {batch_job.get('id')}")
            return batch_job

        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error during batch creation: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"Batch request failed: {str(e)}")
            raise

    def get_batch_status(self, job_id: str) -> dict:
        """Get the status of a batch job."""
        log.info(f"Checking status for batch job: {job_id}")
        try:
            response = httpx.get(
                f"{self.BATCH_ENDPOINT}/{job_id}",
                headers=self._headers,
                proxy=self._proxy,
                timeout=15
            )
            response.raise_for_status()
            status = response.json()
            log.info(f"Batch status: {status.get('status', 'unknown')}")
            return status
        except Exception as e:
            log.error(f"Failed to get batch status: {str(e)}")
            raise

    def download_batch_results(self, job_id: str, output_dir: Path, task_name: str) -> Path:
        """Download the results of a completed batch job."""
        log.info(f"Downloading results for batch job: {job_id}")
        try:
            # 1. Get batch status to find the output file ID
            status = self.get_batch_status(job_id)
            if status.get("status") != "completed":
                raise ValueError(f"Batch job {job_id} is not completed. Current status: {status.get('status')}")

            # Extract output file ID - check both locations based on API version
            output_file_id = None
            if "output_file_id" in status:
                output_file_id = status.get("output_file_id")
            elif "result_files" in status and isinstance(status["result_files"], dict) and "id" in status[
                "result_files"]:
                output_file_id = status["result_files"]["id"]

            if not output_file_id:
                raise ValueError(f"No output file ID found for batch job {job_id}")

            # 2. Get file content
            response = httpx.get(
                f"{self.BASE}/files/{output_file_id}/content",
                headers=self._headers,
                proxy=self._proxy,
                timeout=60
            )
            response.raise_for_status()

            # 3. Count lines → "<task>_<n>.jsonl"
            content_bytes = response.content
            line_count = len(content_bytes.splitlines())

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{task_name}_{line_count}.jsonl"
            output_path.write_bytes(content_bytes)
            log.info(f"Batch results saved to {output_path}")

            return output_path

        except Exception as e:
            log.error(f"Failed to download batch results: {str(e)}")
            raise
