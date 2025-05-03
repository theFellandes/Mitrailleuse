from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    # <- add


class Task(BaseModel):
    user_id: str
    api_name: str
    task_name: str

    # 1️⃣  Default to *now* when not provided
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    status: TaskStatus = TaskStatus.PENDING

    # --------------------------------------------------------------
    @property
    def folder_name(self) -> str:
        # 2️⃣  Guard against None just in case
        t = self.created_at or datetime.utcnow()
        return f"{self.api_name}_{self.task_name}_{t.strftime('%d_%m_%Y_%H%M%S')}"

    def path(self, root: Path) -> Path:
        return root / self.user_id / self.folder_name
