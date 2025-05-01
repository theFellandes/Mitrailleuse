# mitrailleuse/application/services/request_service.py
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .task_service import TaskService
from ...application.ports.api_port import APIPort
from ...application.ports.cache_port import CachePort
from ...domain.models import Task, TaskStatus
from ...config.config import Config
from ...infrastructure.adapters.file_cache_adapter import FileCache
from ...infrastructure.adapters.openai_adapter import OpenAIAdapter
from ...infrastructure.utils.mp_pool import pick_num_processes
from ...infrastructure.logging.logger import get_logger
from ...infrastructure.settings import TASK_ROOT
from ... import mitrailleuse_pb2

log = get_logger(__name__)


class RequestService:
    """Coordinates multiprocessing + caching + API adapter."""

    def __init__(self, api: APIPort, cache: CachePort, config: Config):
        self.api = api
        self.cache = cache
        self.config = config

    # ---------------------------------------------------------- helpers
    def _send_single(self, file_path: Path):
        user_payload = json.loads(file_path.read_text())
        body = self._build_openai_body(user_payload)
        log.info(f"Sending request for {file_path.name}")
        return file_path.name, self.api.send_single(body)

    # ---------------------------------------------------------- main
    def execute(self, task: Task, base_path: Path) -> str | None:
        """Return job_id when batch is launched else None."""
        task.status = TaskStatus.RUNNING
        log.info(f"Executing task: {task.task_name} in {base_path}")

        # Create necessary directories
        inputs_path = base_path / "inputs"
        logs_path = base_path / "logs"
        cache_path = base_path / "cache"

        logs_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)

        files = list(inputs_path.glob("*.json"))
        if not files:
            log.error(f"No input files in {inputs_path}")
            task.status = TaskStatus.FAILED
            return None

        # ---------------- batch mode
        if self._is_batch_enabled(self.config):
            log.info("Using batch mode for request")
            payloads = [
                self._build_openai_body(json.loads(f.read_text()))
                for f in files
            ]

            try:
                job_obj = self.api.send_batch(payloads)
                (base_path / "cache" / "batch_job.json").write_text(
                    json.dumps(job_obj, indent=2))
                task.status = TaskStatus.SUCCESS
                log.info(f"Batch job created with ID: {job_obj.get('id', 'unknown')}")
                return job_obj.get("id")
            except Exception as e:
                log.error(f"Batch request failed: {str(e)}")
                task.status = TaskStatus.FAILED
                return None

        # --------------- single-shot (maybe multiproc)
        log.info("Using direct request mode")
        n_proc = pick_num_processes(self.config)
        log.info(f"Using {n_proc} processes for direct requests")

        try:
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                futures = {pool.submit(self._send_single, f): f for f in files}
                for fut in as_completed(futures):
                    try:
                        fname, resp = fut.result()
                        self.cache.set(fname, resp)
                        (base_path / "outputs" / f"resp_{fname}").write_text(json.dumps(resp, indent=2))
                    except Exception as e:
                        log.error(f"Error processing request: {str(e)}")
                        task.status = TaskStatus.FAILED
                        return None

            self.cache.flush_to_disk()
            task.status = TaskStatus.SUCCESS
            return None
        except Exception as e:
            log.error(f"Error in execute: {str(e)}")
            task.status = TaskStatus.FAILED
            return None

    def _openai_model(self):
        """Safely get the model from either dict-style or object-style config."""
        try:
            # First try object-style access
            return self.config.openai.api_information.model
        except (AttributeError, TypeError):
            # Fall back to dict-style access
            try:
                return self.config["openai"]["api_information"]["model"]
            except (KeyError, TypeError):
                log.error("Failed to retrieve OpenAI model from config")
                return "gpt-4o"  # Default fallback

    def _build_openai_body(self, payload: dict) -> dict:
        """Build the request payload for OpenAI API."""
        model = self._openai_model()

        # Handle dynamic system instruction if configured
        system_content = payload.get("instruction", "")
        if hasattr(self.config, "openai") and hasattr(self.config.openai, "system_instruction"):
            if getattr(self.config.openai.system_instruction, "is_dynamic", False):
                # Use dynamic system prompt if enabled
                system_content = getattr(self.config.openai.system_instruction, "system_prompt", system_content)

        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": payload.get("user_prompt", "")}
            ]
        }

    @staticmethod
    def _is_batch_enabled(cfg_dict_or_obj) -> bool:
        """Return True when openai.batch.is_batch_active == true."""
        try:
            # object-style access (works if nested fields are pydantic models)
            return bool(cfg_dict_or_obj.openai.batch.is_batch_active)
        except AttributeError:
            # dict-style access (what we have after json round-trip)
            try:
                return bool(cfg_dict_or_obj["openai"]["batch"]["is_batch_active"])
            except (KeyError, TypeError):
                return False

    def ExecuteTask(self, request, context):
        task_path = Path(request.task_folder)
        cfg = Config.read(task_path / "config" / "config.json")
        cache = FileCache(task_path / "cache")
        api = OpenAIAdapter(cfg)
        task = TaskService.status_from_path(task_path)

        job_id = RequestService(api, cache, cfg).execute(task)

        return mitrailleuse_pb2.ExecuteTaskResponse(
            status=task.status.value, job_id=job_id or ""
        )
