import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_task_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)