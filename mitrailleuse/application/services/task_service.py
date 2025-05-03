import json
from copy import deepcopy
from pathlib import Path
import shutil
from typing import Tuple
from mitrailleuse.domain.models import Task, TaskStatus
from mitrailleuse.config.config import Config
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.settings import TASK_ROOT, REPO_ROOT, TEMPLATE_CONFIG

# *****  choose a project-wide tasks root you like *****
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
TASK_ROOT = PROJECT_ROOT / "tasks"

log = get_logger(__name__)


class TaskService:
    """Handles task life-cycle, folder provisioning, config persistence."""

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

        # Ensure task_name is set correctly and save config
        if isinstance(cfg, Config):
            # ── live copy (deep-clone dict ➜ update task_name) ───────────────────
            try:
                live_cfg = cfg.copy(update={"task_name": task.task_name})
                config_path = base_path / "config" / "config.json"
                Config.write(live_cfg, base_path / "config" / "config.json")
                log.info(f"Saved task config to {config_path}")
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
