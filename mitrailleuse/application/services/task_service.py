from pathlib import Path
import shutil
from typing import Tuple
from mitrailleuse.domain.models import Task, TaskStatus
from mitrailleuse.config.config import Config
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.settings import TASK_ROOT, REPO_ROOT, TEMPLATE_CONFIG

# *****  choose a project-wide tasks root you like *****
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # repo root
TASK_ROOT    = PROJECT_ROOT / "tasks"

log = get_logger(__name__)


class TaskService:
    """Handles task life-cycle, folder provisioning, config persistence."""

    @staticmethod
    def create_task(user_id: str, api_name: str,
                    task_name: str, cfg: Config) -> Task:
        task = Task(user_id=user_id, api_name=api_name, task_name=task_name)
        base_path = task.path(TASK_ROOT)

        # folders
        for leaf in ("config", "logs", "inputs", "outputs", "cache"):
            (base_path / leaf).mkdir(parents=True, exist_ok=True)

        # 1️⃣  template → config_original.json  (unchanged)
        shutil.copy(TEMPLATE_CONFIG, base_path / "config" / "config_original.json")

        # 2️⃣  live copy → config.json  (task_name UPDATED)
        live_cfg = cfg.model_copy(update={"task_name": task.folder_name})
        Config.write(live_cfg, base_path / "config" / "config.json")

        log.info("Task folder ready at %s", base_path)
        return task

    # ------------------------------------------------------------------------
    @staticmethod
    def status_from_path(task_path: Path) -> Task:
        """Lightweight reconstruction when we only have the folder path."""
        parts       = task_path.parts[-3:]                # user_id / api_... / <date>
        user_id     = parts[0]
        api, *rest  = parts[1].split("_", 1)
        task_name   = rest[0] if rest else "unknown"
        return Task(user_id=user_id, api_name=api, task_name=task_name,
                    created_at=None, status=TaskStatus.PENDING)

    @staticmethod
    def status(task: Task) -> Tuple[str, str]:
        return task.status, str(task.path(TASK_ROOT))
