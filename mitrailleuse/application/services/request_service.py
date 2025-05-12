import datetime as dt
import json
import math
import random
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .task_service import TaskService
from ... import mitrailleuse_pb2
from ...application.ports.api_port import APIPort
from ...application.ports.cache_port import CachePort
from ...config.config import Config
from ...domain.models import Task, TaskStatus
from ...infrastructure.adapters.deepl_adapter import DeepLAdapter
from ...infrastructure.adapters.deepseek_adapter import DeepSeekAdapter
from ...infrastructure.adapters.file_cache_adapter import FileCache
from ...infrastructure.adapters.openai_adapter import OpenAIAdapter
from ...infrastructure.logging.logger import get_logger
from ...infrastructure.utils.mp_pool import pick_num_processes

log = get_logger(__name__)

ADAPTERS = {
    "openai":  OpenAIAdapter,
    "deepseek": DeepSeekAdapter,
    "deepl":    DeepLAdapter,
    # extend here for any future provider
}


class RequestService:
    """Coordinates multiprocessing + caching + API adapter."""

    def __init__(self, api: APIPort, cache: CachePort, config: Config):
        self.api = api
        self.cache = cache
        self.config = config

        # ── derive the task-name once, for all filenames ───────────────────
        # preferred: explicit field in Config
        self.task_name: str | None = getattr(config, "task_name", None)

        # fallback ① – nested under ".task.task_name"
        if not self.task_name:
            self.task_name = (
                getattr(config, "task", {}).get("task_name", None)
                if isinstance(getattr(config, "task", None), dict)
                else None
            )

        # fallback ② – last segment of the working-directory path
        if not self.task_name:
            # assume Config has a "workdir" or similar path attribute
            workdir = Path(getattr(config, "workdir", ".")).resolve()
            self.task_name = workdir.name.split("_", 1)[0] or "task"

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _language_from_filename(p: Path) -> str:
        """
        Detect 2‑letter language code at the start of 'en_input_file.json'.
        Returns 'en' if nothing matches.
        """
        prefix = p.stem.split("_", 1)[0]  # first chunk before underscore
        return prefix.lower() if (len(prefix) == 2 and prefix.isalpha()) else "en"

    def _result_filename(self, input_file: Path, task_name: str, size: int) -> str:
        """
        Build a unique output name for **single-shot** requests:

            <lang>_<task>_<input-stem>_<size>.jsonl

        • lang        → first token of the input file (defaults to 'en')
        • task        → taken from config / function arg
        • input-stem  → 'input_2' from  'input_2.json'
        • size        → object count (1 here, but keep for symmetry)
        """
        lang = self._language_from_filename(input_file)  # defaults to 'en'
        stem = input_file.stem  # e.g. 'input_2'
        return f"{lang}_{task_name}_{stem}_{size}.jsonl"

    def _send_single(self, file_path: Path):
        user_payload = json.loads(file_path.read_text())

        # Choose provider‑specific body
        if isinstance(self.api, DeepLAdapter):
            body_or_payload = user_payload  # DeepL maps internally
        else:
            body_or_payload = self._build_openai_body(user_payload)

        log.info(f"Sending request for {file_path.name}")
        response = self.api.send_single(body_or_payload)

        # Return both the path and response; the parent process will
        # write the file to disk to avoid path‑scope issues inside workers.
        return file_path, response

    @staticmethod
    def _build_jsonl_file(payloads: list[dict], base: Path) -> Path:
        jsonl_path = base / "inputs" / f"inputs_{uuid.uuid4().hex}.jsonl"
        with open(jsonl_path, "w", encoding="utf‑8") as f:
            for row in payloads:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return jsonl_path

    @staticmethod
    def _jsonl_to_json_array(path: Path) -> Path:
        """
        Convert *.jsonl → *.jsonl where the content is a single JSON list.
        Returns the same path so caller code needs no changes.
        """
        tmp_lines = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:                       # skip blank lines
                    tmp_lines.append(json.loads(line))
        # overwrite in-place with the wrapped list
        path.write_text(json.dumps(tmp_lines, ensure_ascii=False, indent=2))
        return path

    @staticmethod
    def _count_items(fp: Path) -> int:
        with fp.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                return len(json.load(f))  # wrapped list
            return sum(1 for _ in f)  # plain JSONL

    def _track_batch_job(self, job_obj, base_path, task_name: str):
        """Track batch job status in a daemon thread."""
        interval = self._get_batch_check_time(self.config)
        log_file_path = base_path / "logs" / "batch.log"
        outputs_path = base_path / "outputs"

        with open(log_file_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"Started tracking batch job {job_obj['id']} at {dt.datetime.now().isoformat()}\n")
            log_f.flush()

            while True:
                try:
                    status = self.api.get_batch_status(job_obj["id"])

                    # Compute progress & ETA when possible
                    # Check in both places where these values might be in the OpenAI API response
                    done = status.get("completed_count", status.get("completed_at", 0))
                    total = status.get("total_count", status.get("total", 1))

                    # Ensure we have valid values for calculations
                    if isinstance(done, str) or done is None:
                        done = 0
                    if isinstance(total, str) or total is None or total == 0:
                        total = 1

                    pct = math.floor(100 * done / total) if total > 0 else 0
                    eta = "n/a"

                    if done and done < total:
                        created_at = status.get("created_at", time.time())
                        if isinstance(created_at, str):
                            # Parse ISO format if needed
                            try:
                                created_at = dt.datetime.fromisoformat(
                                    created_at.replace('Z', '+00:00')).timestamp()
                            except ValueError:
                                created_at = time.time()

                        elapsed = time.time() - created_at
                        eta_secs = elapsed * (total - done) / done if done > 0 else 0
                        eta = str(dt.timedelta(seconds=int(eta_secs)))

                    # Create status snapshot
                    snap = {
                        "status": status.get("status", "unknown"),
                        "done": done,
                        "total": total,
                        "progress": f"{pct}%",
                        "eta": eta,
                        "timestamp": dt.datetime.now().isoformat()
                    }

                    # Log status
                    log_f.write(json.dumps(snap) + "\n")
                    log_f.flush()

                    # Also write to outputs for easy tracking
                    outputs_path.mkdir(parents=True, exist_ok=True)
                    (outputs_path / "batch_status.json").write_text(json.dumps(snap, indent=2))

                    # Check if completed or failed
                    if status.get("status") in {"completed", "failed", "expired", "cancelled"}:
                        log_f.write(
                            f"Batch job {job_obj['id']} finished with status: {status.get('status')}\n")

                        # If completed successfully, try to download results
                        if status.get("status") == "completed":
                            try:

                                log_f.write("Attempting to download batch results...\n")
                                result_path: Path = self.api.download_batch_results(
                                    job_obj["id"], outputs_path, task_name
                                )

                                # 🔄  Convert JSON-Lines → single JSON list
                                self._jsonl_to_json_array(result_path)

                                # 📏  Count items and rename:  <task>_<n>.jsonl
                                size = self._count_items(result_path)

                                final_name = f"{task_name}_{size}.jsonl"
                                final_path = result_path.with_name(final_name)

                                if final_path != result_path:
                                    result_path.rename(final_path)

                                msg = f"Batch results saved to {final_path} ({size} items)"
                                log_f.write(msg + "\n")
                                log.info(msg)
                            except Exception as dl_err:
                                log_f.write(f"Failed to download results: {dl_err}\n")

                        break

                except Exception as loop_err:
                    log_f.write(f"Error checking batch status: {str(loop_err)}\n")
                    log_f.flush()

                time.sleep(interval)

    # ---------------------------------------------------------- main
    def execute(self, task: Task, base_path: Path) -> str | None:
        """Return job_id when batch is launched else None."""
        task.status = TaskStatus.RUNNING
        log.info(f"Executing task: {task.task_name} in {base_path}")

        # Create necessary directories
        inputs_path = base_path / "inputs"
        outputs_path = base_path / "outputs"
        logs_path = base_path / "logs"
        cache_path = base_path / "cache"

        inputs_path.mkdir(parents=True, exist_ok=True)
        outputs_path.mkdir(parents=True, exist_ok=True)
        logs_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)

        files = list(inputs_path.glob("*.json"))
        if self._sampling_enabled(self.config):
            log.info("Sampling enabled, creating JSONL file")
            limit = self._sample_size(self.config)
            if limit and len(files) > limit:
                files = random.sample(files, limit)
                log.info(f"Sampling enabled → taking {limit} of {len(files)} input files")

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

                # 1) persist job‑meta
                cache_file = base_path / "cache" / "batch_job.json"
                cache_file.write_text(json.dumps(job_obj, indent=2))

                # Start the tracking thread
                tracking_thread = threading.Thread(
                    target=self._track_batch_job,
                    args=(job_obj, base_path, self.task_name),  # ← always filled
                    daemon=True
                )
                tracking_thread.start()
                log.info(f"Started batch tracking thread for job {job_obj['id']}")

                task.status = TaskStatus.SUCCESS
                log.info(f"Batch job created with ID: {job_obj['id']}")
                return job_obj["id"]

            except Exception as e:
                log.error(f"Batch request failed: {str(e)}")
                task.status = TaskStatus.FAILED
                return None

        # --------------- single-shot (maybe multiproc)
        log.info("Using direct request mode")
        n_proc = pick_num_processes(self.config)
        log.info(f"Using {n_proc} processes for direct requests")

        try:
            outputs_path.mkdir(parents=True, exist_ok=True)
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                futures = {pool.submit(self._send_single, f): f for f in files}
                for fut in as_completed(futures):
                    try:
                        fname, resp = fut.result()
                        self.cache.set(fname.name, resp)
                        out_name = self._result_filename(fname, self.task_name, 1)
                        output_file = outputs_path / out_name
                        output_file.write_text(json.dumps(resp, indent=2))
                        log.info("Saved response for %s → %s", fname.name, out_name)
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

    @staticmethod
    def _sampling_enabled(cfg) -> bool:
        try:
            return bool(cfg.general.sampling.enable_sampling)
        except AttributeError:
            try:
                return bool(cfg["general"]["sampling"]["enable_sampling"])
            except (KeyError, TypeError):
                return False

    @staticmethod
    def _sample_size(cfg) -> int:
        try:
            return int(cfg.general.sampling.sample_size)
        except AttributeError:
            try:
                return int(cfg["general"]["sampling"]["sample_size"])
            except (KeyError, TypeError, ValueError):
                return 0

    def _openai_model(self):
        """Safely get the model from either dict-style or object-style config."""
        try:
            # First try object-style access
            return (
                self.config.get("deepseek", {})
                .get("api_information", {})
                .get("model", "gpt-4o")
                if isinstance(self.config, dict) else "gpt-4o"
            )
        except (AttributeError, TypeError):
            # Fall back to dict-style access
            try:
                return self.config["openai"]["api_information"]["model"]
            except (KeyError, TypeError):
                log.error("Failed to retrieve OpenAI model from config")
                return "gpt-4o"  # Default fallback

    # ───────────────────────────────────────────────────────────────────
    # Dynamic prompt builder
    # ------------------------------------------------------------------
    def _build_openai_body(self, payload: dict) -> dict:
        """
        Build chat‑completion body for **both OpenAI and DeepSeek** using the
        mapping rules stored in the task’s config:

          config.prompt            → name of user‑prompt field  (default: "user_prompt")
          config.system_instruction.is_dynamic
              false  → system prompt = "You are a helpful assistant"
              true   → payload[config.system_instruction.system_prompt]
                                                    (default: "instructions")
        """

        # ---------------- helpers ----------------
        def cfg_get(path: str, default=None):
            """
            Dot‑path getter that works on either pydantic objects or dicts.
            Looks first under a provider section (openai/deepseek), then top‑level.
            """
            providers = ("openai", "deepseek")  # order of preference
            for root in providers + ("",):
                cur = self.config if root == "" else getattr(self.config, root, None) or \
                                                     (self.config.get(root) if isinstance(self.config, dict) else None)
                if cur is None:
                    continue
                node = cur
                for part in path.split("."):
                    if isinstance(node, dict):
                        node = node.get(part)
                    else:
                        node = getattr(node, part, None)
                    if node is None:
                        break
                if node is not None:
                    return node
            return default

        # ---------------- mapping keys ----------------
        prompt_key = cfg_get("prompt", "user_prompt")
        is_dyn = bool(cfg_get("system_instruction.is_dynamic", False))
        sys_prompt_key = cfg_get("system_instruction.system_prompt", "instructions")

        # ---------------- build message list ----------------
        user_content = payload.get(prompt_key, "")
        if is_dyn:
            system_content = payload.get(sys_prompt_key, "")
        else:
            system_content = "You are a helpful assistant"

        return {
            "model": self._openai_model(),  # same helper as before
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        }

    @staticmethod
    def _get_batch_check_time(cfg):
        """Get the batch check time from config, with fallback to default."""
        try:
            # Try object-style access
            return getattr(cfg.openai.batch, "batch_check_time", 10)
        except (AttributeError, TypeError):
            # Try dict-style access
            try:
                return cfg["openai"]["batch"]["batch_check_time"]
            except (KeyError, TypeError):
                return 10  # Default

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
        """gRPC endpoint implementation."""
        task_path = Path(request.task_folder)
        cfg = Config.read(task_path / "config" / "config.json")
        cache = FileCache(task_path / "cache")
        adapter_cls = ADAPTERS.get(request.api_name.lower(), OpenAIAdapter)
        api = adapter_cls(cfg)
        self.api = api
        task = TaskService.status_from_path(task_path)

        job_id = self.execute(task, task_path)

        return mitrailleuse_pb2.ExecuteTaskResponse(
            status=task.status.value, job_id=job_id or ""
        )
