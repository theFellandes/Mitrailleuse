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
    def _send_single_cached(self, file_path: Path):
        body = json.loads(file_path.read_text())
        key  = str(file_path)

        if self.cache.has(key):
            return self.cache.get(key)

        resp = self.api.send_single(body)
        self.cache.set(key, resp)
        return resp

    # ---------------------------------------------------------- main
    def execute(self, task: Task) -> str | None:
        """Returns batch job id (str) if a batch was started else None."""
        task.status = TaskStatus.RUNNING
        base_path   = task.path(TASK_ROOT)
        inputs_path = base_path / "inputs"
        files: List[Path] = list(inputs_path.glob("*.json"))

        if not files:
            log.warning("No input files found for task %s", task.folder_name)
            task.status = TaskStatus.FAILED
            return None

        # ------------------------- CASE 1:  OpenAI batch
        if self.config.openai.batch.is_batch_active:
            payloads = [json.loads(f.read_text()) for f in files]
            job_obj  = self.api.send_batch(payloads)          # ← single call
            job_id   = job_obj["id"]

            cache_dir = base_path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "batch_job.json").write_text(json.dumps(job_obj, indent=2))

            self.cache.flush_to_disk()
            task.status = TaskStatus.SUCCESS
            log.info("Batch job %s registered for task %s", job_id, task.folder_name)
            return job_id

        # ------------------------- CASE 2:  single-shot (maybe multiproc)
        results = []
        n_proc  = pick_num_processes(self.config)
        with ProcessPoolExecutor(max_workers=n_proc) as pool:
            futs = {pool.submit(self._send_single_cached, f): f for f in files}
            for fut in as_completed(futs):
                results.append(fut.result())

        self.cache.flush_to_disk()
        task.status = TaskStatus.SUCCESS
        log.info("Task %s finished (%d single requests)", task.folder_name, len(files))
        return None

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
