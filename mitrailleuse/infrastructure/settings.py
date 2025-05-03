from pathlib import Path
import os

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[1]  # …/Mitrailleuse/mitrailleuse
REPO_ROOT = PACKAGE_ROOT.parent  # …/Mitrailleuse

# --------------------------------------------------------------------------
#  YOUR decision: keep tasks *inside* the package directory
# --------------------------------------------------------------------------
TASK_ROOT = Path(os.getenv(
    "MITRAILLEUSE_TASK_ROOT",
    PACKAGE_ROOT / "tasks"  # <<<<<<  this is new
))

TEMPLATE_CONFIG = PACKAGE_ROOT / "config" / "config.json"
if not TEMPLATE_CONFIG.exists():
    raise FileNotFoundError(f"Missing template at {TEMPLATE_CONFIG}")
