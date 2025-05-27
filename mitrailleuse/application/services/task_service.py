import json
from copy import deepcopy
from pathlib import Path
import shutil
from typing import Tuple, List, Optional
from mitrailleuse.domain.models import Task, TaskStatus
from mitrailleuse.config.config import Config
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.settings import TASK_ROOT, REPO_ROOT, TEMPLATE_CONFIG
from mitrailleuse.infrastructure.adapters.memory_cache_adapter import MemoryCache

# *****  choose a project-wide tasks root you like *****
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
TASK_ROOT = PROJECT_ROOT / "tasks"

log = get_logger(__name__)


class TaskService:
    """Handles task life-cycle, folder provisioning, config persistence."""

    def __init__(self):
        self.memory_cache = MemoryCache()

    @staticmethod
    def list_available_tasks(user_id: str, task_name: str) -> List[Task]:
        """List all available tasks for a user and task name."""
        tasks = []
        user_tasks_dir = TASK_ROOT / user_id
        
        if not user_tasks_dir.exists():
            return tasks
        
        # Get all task directories matching the pattern
        task_dirs = sorted(user_tasks_dir.glob(f"{task_name}_*"), reverse=True)
        
        for task_dir in task_dirs:
            task = TaskService.status_from_path(task_dir)
            if task:
                tasks.append(task)
        
        return tasks

    @staticmethod
    def get_task_by_path(task_path: Path) -> Optional[Task]:
        """Get task information from a specific path."""
        if not task_path.exists():
            return None
        return TaskService.status_from_path(task_path)

    @staticmethod
    def create_task(user_id: str, api_name: str,
                    task_name: str, cfg: Config) -> Task:
        task = Task(user_id=user_id, api_name=api_name, task_name=task_name)
        base_path = task.path(TASK_ROOT)
        log.info(f"Creating new task at {base_path}")

        # Create all required directories
        for leaf in ("config", "logs", "inputs", "outputs", "cache"):
            (base_path / leaf).mkdir(parents=True, exist_ok=True)

        # Save original template config
        if TEMPLATE_CONFIG.exists():
            shutil.copy(TEMPLATE_CONFIG, base_path / "config" / "config_original.json")
            log.info(f"Copied template config from {TEMPLATE_CONFIG}")

        # Validate and ensure required fields are present
        if isinstance(cfg, Config):
            # Get the API section
            api_section = getattr(cfg, api_name.lower(), None)
            if not api_section:
                raise ValueError(f"API section '{api_name}' not found in config")

            # Ensure prompt field is set in the API section
            if not hasattr(api_section, "prompt") or not api_section.prompt:
                log.warning(f"No prompt field specified in {api_name} config, using default 'input_text'")
                setattr(api_section, "prompt", "input_text")

            # Ensure system instruction settings are valid
            if not hasattr(api_section, "system_instruction"):
                setattr(api_section, "system_instruction", {"is_dynamic": False, "system_prompt": ""})
            elif not isinstance(api_section.system_instruction, dict):
                setattr(api_section, "system_instruction", {"is_dynamic": False, "system_prompt": ""})

            # Add cache configuration
            if not hasattr(cfg, "cache"):
                cfg.cache = {
                    "memory_cache_enabled": False,  # Default to disabled
                    "memory_cache_ttl": 3600  # 1 hour default TTL
                }

            # Ensure task_name is set correctly and save config
            try:
                # Create a filtered config with only the relevant API section
                live_cfg = Config.create_filtered_config(cfg, task.api_name, task.user_id)
                config_path = base_path / "config" / "config.json"
                Config.write(live_cfg, base_path / "config" / "config.json")
                log.info(f"Saved task config to {config_path} with only {task.api_name} section")
            except Exception as e:
                log.error(f"Failed to save config: {str(e)}")
                # Use simplified approach as fallback
                Config.write(cfg, base_path / "config" / "config.json")
        else:
            log.error(f"Invalid config type: {type(cfg)}")

        return task

    # ------------------------------------------------------------------------
    @staticmethod
    def status_from_path(task_path: Path) -> Task:
        """Lightweight reconstruction when we only have the folder path."""
        parts = task_path.parts[-3:] if len(task_path.parts) >= 3 else ["unknown", "unknown", "unknown"]
        user_id = parts[0]
        api, *rest = parts[1].split("_", 1) if "_" in parts[1] else (parts[1], "")
        task_name = rest[0] if rest else "unknown"

        # Check for status file
        status_file = task_path / "status.json"
        status = TaskStatus.PENDING

        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text())
                status = TaskStatus(status_data.get("status", "pending"))
            except Exception:
                log.warning(f"Could not read status file at {status_file}")

        return Task(user_id=user_id, api_name=api, task_name=task_name,
                    created_at=None, status=status)

    @staticmethod
    def status(task: Task) -> Tuple[str, str]:
        """Return task status and path."""
        path = task.path(TASK_ROOT)

        # Save status to file for persistence
        try:
            status_file = path / "status.json"
            status_file.write_text(json.dumps({"status": task.status.value}))
        except Exception as e:
            log.warning(f"Failed to save status file: {str(e)}")

        return task.status.value, str(path)
