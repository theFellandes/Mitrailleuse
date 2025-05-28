import datetime as dt
import json
import math
import random
import asyncio
import time
import uuid
import shutil
from pathlib import Path
import multiprocessing
from typing import Union, List, Dict
import logging
from datetime import datetime
import grpc

from mitrailleuse import mitrailleuse_pb2
from mitrailleuse.application.services.task_service import TaskService, TASK_ROOT
from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.application.ports.cache_port import CachePort
from mitrailleuse.infrastructure.utils.cache_manager import CacheManager
from mitrailleuse.config.config import Config
from mitrailleuse.domain.models import Task, TaskStatus
from mitrailleuse.infrastructure.adapters.deepl_adapter import DeepLAdapter
from mitrailleuse.infrastructure.adapters.deepseek_adapter import DeepSeekAdapter
from mitrailleuse.infrastructure.adapters.file_cache_adapter import FileCache
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
from mitrailleuse.infrastructure.adapters.memory_cache_adapter import MemoryCache
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.utils.mp_pool import pick_num_processes
from mitrailleuse.infrastructure.utils.file_converter import FileConverter
from mitrailleuse.infrastructure.utils.file_flattener import FileFlattener
from mitrailleuse.infrastructure.utils.similarity_checker import SimilarityChecker
from mitrailleuse.scripts.format_response import ResponseFormatter

ADAPTERS = {
    "openai": OpenAIAdapter,
    "deepseek": DeepSeekAdapter,
    "deepl": DeepLAdapter,
    # extend here for any future provider
}


class RequestService:
    """Coordinates multiprocessing + caching + API adapter."""

    def __init__(self, api: APIPort, cache: CachePort, config: Config):
        self.api = api
        self.cache = cache
        self.memory_cache = MemoryCache()  # Add in-memory cache
        self.config = config
        self.similarity_checker = None  # Will be initialized in execute()
        self.deepl_client = None  # Will be initialized if needed

        # ── derive the task-name once, for all filenames ───────────────────
        self.task_name: str | None = getattr(config, "task_name", None)

        if not self.task_name:
            self.task_name = (
                getattr(config, "task", {}).get("task_name", None)
                if isinstance(getattr(config, "task", None), dict)
                else None
            )

        if not self.task_name:
            workdir = Path(getattr(config, "workdir", ".")).resolve()
            self.task_name = workdir.name.split("_", 1)[0] or "task"

        # Initialize task-specific logger
        self.log = self._setup_task_logger()

        # Initialize DeepL client if needed
        if isinstance(self.api, DeepLAdapter):
            self.deepl_client = self.api

    def _setup_task_logger(self):
        """Set up task-specific logging."""
        # Get task directory from config
        task_dir = Path(getattr(self.config, "workdir", ".")).resolve()
        log_dir = task_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a new log file for this task
        log_file = log_dir / f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Configure task-specific logger
        logger = logging.getLogger(f"task_{self.task_name}")
        logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

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

    def _backup_input_file(self, input_file: Path, base_path: Path) -> None:
        """Backup the input file to the backup directory."""
        backup_dir = base_path / "inputs" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{input_file.stem}_{timestamp}{input_file.suffix}"
        shutil.copy2(input_file, backup_file)
        self.log.info(f"Backed up input file to: {backup_file}")

    def _convert_to_jsonl(self, input_file: Path, base_path: Path) -> Path:
        """Convert JSON file to JSONL format if needed and backup original."""
        # Backup the original file
        self._backup_input_file(input_file, base_path)

        if input_file.suffix == '.jsonl':
            return input_file

        output_file = input_file.with_suffix('.jsonl')
        with open(input_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        self.log.info(f"Converted {input_file} to JSONL format")
        return output_file

    def _convert_to_json(self, input_file: Path, base_path: Path) -> Path:
        """Convert JSONL file to JSON format if needed and backup original."""
        # Backup the original file
        self._backup_input_file(input_file, base_path)

        if input_file.suffix == '.json':
            return input_file

        output_file = input_file.with_suffix('.json')
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.log.info(f"Converted {input_file} to JSON format")
        return output_file

    async def _send_single(self, file_path: Path, base_path: Path):
        """Process a single file with backup and format conversion."""
        try:
            # Backup original file
            original_dir = base_path / "inputs" / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, original_dir / file_path.name)

            # Convert to JSONL for processing
            jsonl_file = FileConverter.convert_to_jsonl(file_path, base_path)

            # Read all lines from the JSONL file
            with open(jsonl_file, 'r') as f:
                items = [json.loads(line) for line in f if line.strip()]

            results = []
            for item in items:
                # Choose provider-specific body
                if isinstance(self.api, DeepLAdapter):
                    body_or_payload = item
                else:
                    body_or_payload = self._build_openai_body(item)

                # Check cache first
                cache_key = f"{file_path.name}_{len(results)}"
                try:
                    cached_response = await self.cache.get(cache_key)
                    if cached_response is not None:
                        results.append(cached_response)
                        continue
                except Exception as cache_err:
                    self.log.debug(f"Cache miss for key {cache_key}: {str(cache_err)}")

                # If not in cache, send request
                try:
                    self.log.info(f"Sending request for item in {file_path.name}")
                    response = await self.api.send_single(body_or_payload)

                    # Convert response to dict for JSON serialization
                    if hasattr(response, 'model_dump'):
                        response = response.model_dump()

                    # Check similarity if enabled
                    if self.similarity_checker and self._check_similarity_enabled(self.config):
                        is_similar, similarity = self.similarity_checker.check_similarity(response)
                        if is_similar and similarity > 0.0:  # Only apply cooldown if actually similar
                            cooldown_time = self.similarity_checker.get_cooldown_time()
                            self.log.warning(
                                f"Similar response detected (similarity: {similarity:.2f}). "
                                f"Applying cooldown of {cooldown_time} seconds."
                            )
                            await asyncio.sleep(cooldown_time)

                    # Store in cache
                    try:
                        await self.cache.set(cache_key, response)
                    except Exception as cache_err:
                        self.log.warning(f"Failed to cache response for {cache_key}: {str(cache_err)}")

                    results.append(response)
                except Exception as e:
                    self.log.error(f"Error sending request for {file_path.name}: {str(e)}")
                    raise

            # Save responses
            output_dir = base_path / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save raw response
            raw_output = output_dir / f"{file_path.stem}_raw_response.json"
            with open(raw_output, 'w') as f:
                json.dump(results, f, indent=2)

            # Format responses using ResponseFormatter
            try:
                # Get user_id from the task path (parent directory of base_path)
                user_id = base_path.parent.name
                task_name = base_path.name

                # Read the input file to get the original prompts
                input_file = base_path / "inputs" / file_path.name
                with open(input_file, 'r') as f:
                    input_data = json.load(f)
                if not isinstance(input_data, list):
                    input_data = [input_data]

                # Get config values for prompt fields
                prompt_key = self.config.openai.prompt if hasattr(self.config, 'openai') else "input_text"
                is_dynamic = getattr(self.config.openai.system_instruction, 'is_dynamic', False) if hasattr(self.config,
                                                                                                            'openai') else False
                sys_prompt_key = getattr(self.config.openai.system_instruction, 'system_prompt',
                                         'instructions') if hasattr(self.config, 'openai') else 'instructions'

                # Format the responses
                formatted_results = []
                for i, (input_item, response) in enumerate(zip(input_data, results)):
                    try:
                        # Get user content from input
                        if isinstance(input_item, str):
                            user_content = input_item
                        elif isinstance(input_item, dict):
                            if prompt_key in input_item:
                                user_content = input_item[prompt_key]
                            elif len(input_item) == 1:
                                user_content = next(iter(input_item.values()))
                            else:
                                user_content = str(input_item)
                        else:
                            user_content = str(input_item)

                        # Get system content
                        system_content = input_item.get(sys_prompt_key,
                                                        "You are a helpful assistant") if is_dynamic else "You are a helpful assistant"

                        # Get assistant content from response
                        if isinstance(response, dict):
                            if "choices" in response:
                                assistant_content = response["choices"][0]["message"]["content"]
                            elif "translated_text" in response:
                                assistant_content = response["translated_text"]
                            else:
                                assistant_content = str(response)
                        else:
                            assistant_content = str(response)

                        formatted_results.append({
                            "system": system_content,
                            "user": user_content,
                            "assistant": assistant_content
                        })
                    except Exception as e:
                        self.log.error(f"Error formatting response {i}: {str(e)}")
                        formatted_results.append({
                            "system": "error",
                            "user": "error",
                            "assistant": f"Error formatting response: {str(e)}"
                        })

                # Save formatted response
                formatted_output = output_dir / f"{file_path.stem}_formatted_response.json"
                with open(formatted_output, 'w') as f:
                    json.dump(formatted_results, f, indent=2)

                # Save parsed response (content only)
                parsed_output = output_dir / f"parsed_{file_path.stem}_response.jsonl"
                with open(parsed_output, 'w') as f:
                    for result in formatted_results:
                        f.write(json.dumps({"content": result["assistant"]}) + '\n')

                self.log.info(f"Formatted and parsed responses for {file_path.name}")
            except Exception as e:
                self.log.error(f"Error formatting responses: {str(e)}")
                raise

            return file_path, results
        except Exception as e:
            self.log.error(f"Error in _send_single for {file_path}: {str(e)}")
            raise

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
                if line:  # skip blank lines
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

    async def _track_batch_job(self, job_obj, base_path, task_name: str):
        """Track batch job status in a background task."""
        interval = self._get_batch_check_time(self.config)
        log_file_path = base_path / "logs" / "batch.log"
        outputs_path = base_path / "outputs"

        with open(log_file_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"Started tracking batch job {job_obj['id']} at {dt.datetime.now().isoformat()}\n")
            log_f.flush()

            while True:
                try:
                    status = await self.api.get_batch_status(job_obj["id"])

                    # Compute progress & ETA when possible
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
                                result_path: Path = await self.api.download_batch_results(
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
                                self.log.info(msg)
                            except Exception as dl_err:
                                log_f.write(f"Failed to download results: {dl_err}\n")

                        break

                except Exception as loop_err:
                    log_f.write(f"Error checking batch status: {str(loop_err)}\n")
                    log_f.flush()

                await asyncio.sleep(interval)

    # ---------------------------------------------------------- main
    def _calculate_process_cap(self, cfg) -> int:
        """Calculate the maximum number of processes to use."""
        try:
            # Get system CPU count
            cpu_count = multiprocessing.cpu_count()

            # Get process cap percentage from config
            cap_percentage = getattr(cfg.general, "process_cap_percentage", 75)
            base_cap = max(1, math.floor(cpu_count * (cap_percentage / 100)))

            # Get config override if exists
            config_cap = getattr(cfg.general, "num_processes", None)
            if config_cap is not None:
                return min(config_cap, base_cap)

            return base_cap
        except (AttributeError, TypeError):
            # Fallback to dict-style access
            try:
                cap_percentage = cfg["general"]["process_cap_percentage"]
                base_cap = max(1, math.floor(cpu_count * (cap_percentage / 100)))
                config_cap = cfg["general"].get("num_processes")
                if config_cap is not None:
                    return min(config_cap, base_cap)
                return base_cap
            except (KeyError, TypeError):
                return max(1, math.floor(cpu_count * 0.75))  # Default to 75%

    def _should_combine_batches(self, cfg) -> bool:
        """Check if batches should be combined based on config."""
        try:
            # Try object-style access
            return bool(cfg.openai.batch.combine_batches)
        except (AttributeError, TypeError):
            # Try dict-style access
            try:
                return bool(cfg["openai"]["batch"]["combine_batches"])
            except (KeyError, TypeError):
                return False

    async def execute(self, task: Task, base_path: Path) -> str | None:
        """Return job_id when batch is launched else None."""
        task.status = TaskStatus.RUNNING
        self.log.info(f"Executing task: {task.task_name} in {base_path}")

        # Initialize similarity checker if enabled
        if self._check_similarity_enabled(self.config):
            self.similarity_checker = SimilarityChecker(base_path, self.config)
            self.log.info("Similarity checking enabled")

        # Create necessary directories
        FileConverter.ensure_directory_structure(base_path)
        inputs_path = base_path / "inputs"
        outputs_path = base_path / "outputs"
        logs_path = base_path / "logs"
        cache_path = base_path / "cache"

        # Get all JSON and JSONL files
        files = list(inputs_path.glob("*.json")) + list(inputs_path.glob("*.jsonl"))

        if self._sampling_enabled(self.config):
            self.log.info("Sampling enabled, creating JSONL file")
            limit = self._sample_size(self.config)
            if limit and len(files) > limit:
                files = random.sample(files, limit)
                self.log.info(f"Sampling enabled → taking {limit} of {len(files)} input files")

        if not files:
            self.log.error(f"No input files in {inputs_path}")
            task.status = TaskStatus.FAILED
            return None

        # Calculate number of concurrent tasks
        n_tasks = min(len(files), self._calculate_process_cap(self.config))
        self.log.info(f"Using {n_tasks} concurrent tasks for processing")

        try:
            outputs_path.mkdir(parents=True, exist_ok=True)

            # Process files concurrently
            tasks = []
            for f in files:
                task = asyncio.create_task(self._send_single(f, base_path))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.log.error(f"Error processing file: {str(result)}")
                    task.status = TaskStatus.FAILED
                    return None

                if result is None:
                    self.log.error("Received None result from _send_single")
                    task.status = TaskStatus.FAILED
                    return None

                try:
                    fname, responses = result
                    if responses is None:
                        self.log.error(f"Received None responses for {fname}")
                        task.status = TaskStatus.FAILED
                        return None

                    # Save each response separately
                    for i, response in enumerate(responses):
                        if response is None:
                            self.log.error(f"Received None response at index {i} for {fname}")
                            continue

                        out_name = self._result_filename(fname, self.task_name, i + 1)
                        output_file = outputs_path / out_name
                        output_file.write_text(json.dumps(response, indent=2))
                        self.log.info(f"Saved response {i + 1} for {fname.name} → {out_name}")
                except Exception as e:
                    self.log.error(f"Error processing result: {str(e)}")
                    task.status = TaskStatus.FAILED
                    return None

            try:
                await self.cache.flush_to_disk()
            except Exception as e:
                self.log.warning(f"Failed to flush cache to disk: {str(e)}")

            task.status = TaskStatus.SUCCESS
            return None
        except Exception as e:
            self.log.error(f"Error in execute: {str(e)}")
            task.status = TaskStatus.FAILED
            return None
        finally:
            # Clean up similarity checker
            if self.similarity_checker:
                self.similarity_checker.close()

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
                self.log.error("Failed to retrieve OpenAI model from config")
                return "gpt-4o"  # Default fallback

    # ───────────────────────────────────────────────────────────────────
    # Dynamic prompt builder
    # ------------------------------------------------------------------
    def _build_openai_body(self, payload: dict) -> dict:
        """
        Build chat-completion body for both OpenAI and DeepSeek using the
        mapping rules stored in the task's config.

        Handles multiple formats:
        1. Direct prompt field: {"input_text": "..."}
        2. Nested prompts: {"prompts": [{"input_text": "..."}]}
        3. Custom field names from config
        """

        # ---------------- helpers ----------------
        def cfg_get(path: str, default=None):
            """
            Dot-path getter that works on either pydantic objects or dicts.
            Looks first under a provider section (openai/deepseek), then top-level.
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

        def find_prompt_content(data: dict) -> tuple[str, str]:
            """
            Find prompt content and system instruction in various formats.
            Returns (prompt_content, system_content)
            """
            # Get config keys
            prompt_key = cfg_get("prompt", "input_text")
            is_dyn = bool(cfg_get("system_instruction.is_dynamic", False))
            sys_prompt_key = cfg_get("system_instruction.system_prompt", "instructions")

            # Try different formats
            if "prompts" in data and isinstance(data["prompts"], list):
                # Handle nested prompts format
                if not data["prompts"]:
                    raise ValueError("Empty prompts list")
                prompt_data = data["prompts"][0]  # Take first prompt
                user_content = prompt_data.get(prompt_key)
                system_content = prompt_data.get(sys_prompt_key) if is_dyn else "You are a helpful assistant"
            else:
                # Handle direct format
                user_content = data.get(prompt_key)
                system_content = data.get(sys_prompt_key) if is_dyn else "You are a helpful assistant"

            if not user_content:
                raise ValueError(f"Required prompt field '{prompt_key}' not found in input")

            if is_dyn and not system_content:
                self.log.warning(f"Dynamic system prompt field '{sys_prompt_key}' not found, using default")
                system_content = "You are a helpful assistant"

            return user_content, system_content

        try:
            # Find prompt content
            user_content, system_content = find_prompt_content(payload)

            return {
                "model": self._openai_model(),
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            }
        except Exception as e:
            self.log.error(f"Error building OpenAI body: {str(e)}")
            raise

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

    @staticmethod
    def _check_similarity_enabled(cfg) -> bool:
        """Check if similarity checking is enabled in config."""
        try:
            return bool(cfg.general.similarity_check.enabled)
        except AttributeError:
            try:
                return bool(cfg["general"]["similarity_check"]["enabled"])
            except (KeyError, TypeError):
                return False

    async def ExecuteTask(self, request, context):
        """gRPC endpoint implementation."""
        try:
            task_path = Path(request.task_folder)
            if not task_path.exists():
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task folder not found: {request.task_folder}")
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Read config from task directory
            config_path = task_path / "config" / "config.json"
            if not config_path.exists():
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Config file not found: {config_path}")
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Read and parse config
            with open(config_path, 'r') as f:
                cfg = json.load(f)

            # Ensure batch configuration is properly set
            if "openai" not in cfg:
                cfg["openai"] = {}
            if "batch" not in cfg["openai"]:
                cfg["openai"]["batch"] = {}
            if "completion_window" not in cfg["openai"]["batch"]:
                cfg["openai"]["batch"]["completion_window"] = "24h"  # Default 24 hours in correct format
            if "batch_check_time" not in cfg["openai"]["batch"]:
                cfg["openai"]["batch"]["batch_check_time"] = 5  # Default 5 seconds

            # Get task status
            task = TaskService.status_from_path(task_path)

            # Check if batching is enabled and batch_size is set
            batch_config = cfg.get("openai", {}).get("batch", {})
            if batch_config.get("is_batch_active", False) and "batch_size" in batch_config:
                self.log.info("Batch processing enabled")
                batch_size = batch_config["batch_size"]
                self.log.info(f"Using batch size: {batch_size}")

                # Get all input files
                inputs_path = task_path / "inputs"
                input_files = list(inputs_path.glob("*.json")) + list(inputs_path.glob("*.jsonl"))
                
                if not input_files:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("No input files found")
                    return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

                # Process each input file
                for input_file in input_files:
                    try:
                        self.log.info(f"Processing input file: {input_file}")
                        
                        # Convert to JSONL if needed
                        jsonl_file = FileConverter.convert_to_jsonl(input_file, task_path)
                        
                        # Split into batches
                        batch_files = self._split_into_batches(jsonl_file, batch_size)
                        self.log.info(f"Created {len(batch_files)} batch files")
                        
                        # Process each batch
                        for batch_file in batch_files:
                            try:
                                self.log.info(f"Processing batch file: {batch_file}")
                                
                                # Send batch request using OpenAI's batch API
                                batch_response = await self.api.send_file_batch(batch_file)
                                
                                # Check if batch was successful
                                if isinstance(batch_response, dict) and batch_response.get("choices", [{}])[0].get("message", {}).get("content", "").startswith("Batch job failed"):
                                    raise Exception(batch_response["choices"][0]["message"]["content"])
                                
                                # Save batch results
                                output_dir = task_path / "outputs"
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Save raw response
                                raw_output = output_dir / f"{batch_file.stem}_batch_response.json"
                                with open(raw_output, 'w') as f:
                                    json.dump(batch_response, f, indent=2)
                                
                                # Format response using ResponseFormatter
                                formatter = ResponseFormatter(
                                    user_id=task_path.parent.name,  # Get user_id from parent directory
                                    task_name=task_path.name,  # Get task_name from directory name
                                    task_path=task_path
                                )
                                
                                # Format and save the response
                                formatted_results = formatter.format_batch_response(raw_output, output_dir / f"{batch_file.stem}_formatted_response.json")
                                
                                # Save parsed response (content only)
                                parsed_output = output_dir / f"parsed_{batch_file.stem}_response.jsonl"
                                with open(parsed_output, 'w') as f:
                                    for result in formatted_results:
                                        f.write(json.dumps({"content": result["assistant"]}) + '\n')

                                # Clean up batch file
                                if batch_file.exists():
                                    batch_file.unlink()
                                
                            except Exception as e:
                                self.log.error(f"Error processing batch {batch_file}: {str(e)}")
                                raise
                                
                    except Exception as e:
                        self.log.error(f"Error processing input file {input_file}: {str(e)}")
                        raise
                        
                task.status = TaskStatus.SUCCESS
                return mitrailleuse_pb2.ExecuteTaskResponse(
                    status=str(task.status.value),
                    job_id=""
                )
            else:
                # Handle single request processing
                job_id = await self.execute(task, task_path)
                return mitrailleuse_pb2.ExecuteTaskResponse(
                    status=str(task.status.value),
                    job_id=str(job_id) if job_id else ""
                )
                
        except Exception as e:
            self.log.error(f"Error in ExecuteTask: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

    def _format_and_save_response(self, response: Dict, output_dir: Path, stem: str) -> None:
        """Format and save the response in various formats."""
        try:
            # Save formatted response
            formatted_output = output_dir / f"{stem}_formatted.json"
            with open(formatted_output, 'w') as f:
                json.dump(response, f, indent=2)

            # Save parsed response (content only)
            parsed_output = output_dir / f"parsed_{stem}_response.jsonl"
            with open(parsed_output, 'w') as f:
                if isinstance(response, dict):
                    choices = response.get("choices", [])
                    for choice in choices:
                        content = choice.get("message", {}).get("content", "")
                        f.write(json.dumps({"content": content}) + '\n')
                else:
                    f.write(json.dumps({"content": str(response)}) + '\n')

        except Exception as e:
            self.log.error(f"Error formatting response: {str(e)}")
            raise

    async def ListTasks(self, request, context):
        """List available tasks for a user and task name."""
        try:
            tasks = TaskService.list_available_tasks(request.user_id, request.task_name)
            return mitrailleuse_pb2.ListTasksResponse(
                tasks=[
                    mitrailleuse_pb2.TaskInfo(
                        user_id=task.user_id,
                        api_name=task.api_name,
                        task_name=task.task_name,
                        status=task.status.value,
                        path=str(task.path(TASK_ROOT))
                    )
                    for task in tasks
                ]
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mitrailleuse_pb2.ListTasksResponse()

    async def GetTaskByPath(self, request, context):
        """Get task information from a specific path."""
        try:
            task = TaskService.get_task_by_path(Path(request.task_path))
            if not task:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task not found at path: {request.task_path}")
                return mitrailleuse_pb2.TaskInfo()

            return mitrailleuse_pb2.TaskInfo(
                user_id=task.user_id,
                api_name=task.api_name,
                task_name=task.task_name,
                status=task.status.value,
                path=str(task.path(TASK_ROOT))
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mitrailleuse_pb2.TaskInfo()

    def close(self) -> None:
        """Close the client and its resources."""
        if hasattr(self.cache, 'close'):
            self.cache.close()
        if hasattr(self.similarity_checker, 'close'):
            self.similarity_checker.close()
        # Don't try to close the logger as it doesn't have a close method

    def _should_use_file_batch(self, cfg) -> bool:
        """Check if file batch mode should be used."""
        try:
            # Try object-style access
            return bool(cfg.openai.batch.file_batch)
        except (AttributeError, TypeError):
            # Try dict-style access
            try:
                return bool(cfg["openai"]["batch"]["file_batch"])
            except (KeyError, TypeError):
                return False

    def _split_into_batches(self, jsonl_file: Path, batch_size: int) -> List[Path]:
        """Split a JSONL file into smaller batch files based on batch_size."""
        try:
            batch_files = []
            inputs_dir = jsonl_file.parent  # Use the inputs directory

            # Read all lines from the JSONL file
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            # Calculate number of batches needed
            total_items = len(lines)
            num_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division
            
            self.log.info(f"Splitting {total_items} items into {num_batches} batches of size {batch_size}")

            # Create batch files
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_items)
                batch_lines = lines[start_idx:end_idx]
                
                # Create batch file in inputs directory
                batch_file = inputs_dir / f"batch_{i:04d}.jsonl"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(batch_lines))
                
                self.log.info(f"Created batch file {batch_file.name} with {len(batch_lines)} items")
                batch_files.append(batch_file)

            return batch_files

        except Exception as e:
            self.log.error(f"Error splitting into batches: {str(e)}")
            raise

    async def _process_batch_file(self, input_file: Union[str, Path], service: str, base_path: Union[str, Path],
                                  config: Dict) -> List[Dict]:
        """Process a batch file and return the results."""
        try:
            # Convert string paths back to Path objects
            input_file = Path(input_file)
            base_path = Path(base_path)

            # Initialize components for this process
            cache_manager = CacheManager(base_path, config)

            # Backup original file
            original_dir = base_path / "inputs" / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_file, original_dir / input_file.name)

            # Convert to JSONL and process
            jsonl_file = FileConverter.convert_to_jsonl(input_file, base_path)
            batch_size = config["openai"]["batch"]["batch_size"] if service == "openai" else config["deepl"].get(
                "batch_size", 50)
            batch_files = self._split_into_batches(jsonl_file, batch_size)

            all_results = []
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = [json.loads(line) for line in f if line.strip()]

                    # Send batch request
                    if service == "openai":
                        if self._should_use_file_batch(config):
                            # Use file batch API
                            batch_response = await self.api.send_file_batch(batch_file)

                            # Check if batch was successful
                            if isinstance(batch_response, dict) and batch_response.get("choices", [{}])[0].get(
                                    "message", {}).get("content", "").startswith("Batch job failed"):
                                raise Exception(batch_response["choices"][0]["message"]["content"])

                            batch_results = batch_response
                        else:
                            # Use normal API with batching
                            batch_response = await self.api.send_batch(batch_data)

                            # Handle different response types
                            if isinstance(batch_response, dict):
                                batch_results = batch_response
                            elif hasattr(batch_response, 'model_dump'):
                                batch_results = batch_response.model_dump()
                            elif hasattr(batch_response, 'dict'):
                                batch_results = batch_response.dict()
                            elif hasattr(batch_response, 'json'):
                                batch_results = batch_response.json()
                            else:
                                # Try to convert to dict if possible
                                try:
                                    batch_results = json.loads(str(batch_response))
                                except (json.JSONDecodeError, TypeError):
                                    batch_results = {
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": str(batch_response)
                                                }
                                            }
                                        ]
                                    }
                    elif service == "deepl":
                        batch_results = await self.deepl_client.translate_text(
                            [item.get("text", "") for item in batch_data],
                            target_lang=config["deepl"]["target_lang"]
                        )
                        batch_results = {
                            "responses": [
                                {
                                    "translated_text": t.text,
                                    "detected_source_lang": t.detected_source_lang
                                }
                                for t in batch_results
                            ]
                        }

                    # Save batch results
                    output_dir = base_path / "outputs"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save raw response
                    raw_output = output_dir / f"{batch_file.stem}_batch_response.json"
                    with open(raw_output, 'w') as f:
                        json.dump(batch_results, f, indent=2)

                    # Format response using ResponseFormatter
                    formatter = ResponseFormatter(
                        user_id=base_path.parent.name,  # Get user_id from parent directory
                        task_name=base_path.name,  # Get task_name from directory name
                        task_path=base_path
                    )
                    
                    # Format and save the response
                    formatted_results = formatter.format_batch_response(raw_output, output_dir / f"{batch_file.stem}_formatted_response.json")
                    
                    # Save parsed response (content only)
                    parsed_output = output_dir / f"parsed_{batch_file.stem}_response.jsonl"
                    with open(parsed_output, 'w') as f:
                        for result in formatted_results:
                            f.write(json.dumps({"content": result["assistant"]}) + '\n')

                    # Add batch results to all results
                    if service == "openai":
                        if isinstance(batch_results, dict):
                            all_results.extend(batch_results.get("choices", []))
                        else:
                            all_results.append({"message": {"content": str(batch_results)}})
                    else:
                        all_results.extend(batch_results.get("responses", []))

                except Exception as e:
                    self.log.error(f"Error processing batch {batch_file} with {service}: {str(e)}")
                    raise

            # Combine results if configured
            if self._should_combine_batches(config):
                combined_output = base_path / "outputs" / f"{input_file.stem}_combined_response.jsonl"
                with open(combined_output, 'w') as f:
                    for response in all_results:
                        if isinstance(response, dict):
                            if service == "openai":
                                content = response.get("message", {}).get("content", "")
                            else:
                                content = response.get("translated_text", "")
                            f.write(json.dumps({"content": content}) + '\n')

            return all_results

        except Exception as e:
            self.log.error(f"Error processing batch {input_file.name}: {str(e)}")
            return [{"error": str(e)}]
        finally:
            # Clean up
            cache_manager.close()
