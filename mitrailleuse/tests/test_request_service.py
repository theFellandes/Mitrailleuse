import pytest
import asyncio
from pathlib import Path
from mitrailleuse.application.services.request_service import RequestService
from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.application.ports.cache_port import CachePort
from mitrailleuse.config.config import Config, GeneralConfig, Sampling, GeneralLoggingConfig, DBConfig, SimilarityCheck
from mitrailleuse.domain.models import TaskStatus
from mitrailleuse.application.services.task_service import TaskService

def minimal_config():
    openai_dict = {
        "prompt": "input_text",
        "system_instruction": {"is_dynamic": False, "system_prompt": ""},
        "api_key": "dummy",
        "api_information": {"model": "gpt-4", "setting": {"temperature": 0.7, "max_tokens": 10}},
        "batch": {"is_batch_active": False, "batch_size": 1, "batch_check_time": 10, "combine_batches": False, "file_batch": False},
        "bulk_save": 1,
        "sleep_time": 0,
        "proxies": {"proxies_enabled": False, "http": "", "https": ""}
    }
    cfg = Config(
        task_name="test_task",
        user_id="user1",
        openai=openai_dict,
        general=GeneralConfig(
            verbose=True,
            sampling=Sampling(enable_sampling=False, sample_size=1),
            logs=GeneralLoggingConfig(
                log_file="log.log", log_to_file=False, log_to_db=False, log_buffer_size=10
            ),
            multiprocessing_enabled=False,
            num_processes=1,
            db=DBConfig(postgres={}),
            similarity_check=SimilarityCheck()
        )
    )
    return cfg

class DummyAPI(APIPort):
    async def ping(self): return True
    async def send_single(self, payload): return {"choices": [{"message": {"content": "ok"}}]}
    async def send_batch(self, payloads): return [{"choices": [{"message": {"content": "ok"}}]} for _ in payloads]
    async def send_file_batch(self, file_path): return {"choices": [{"message": {"content": "ok"}}]}
    async def get_batch_status(self, job_id): return {"status": "completed", "completed_count": 1, "total_count": 1}
    async def download_batch_results(self, job_id, output_dir, task_name): return output_dir / f"{task_name}_batch_results.jsonl"
    async def close(self): pass

class DummyCache(CachePort):
    def has(self, key): return False
    def get(self, key): return None
    def set(self, key, value): pass
    def flush_to_disk(self): pass

@pytest.fixture
def dummy_config(tmp_path):
    cfg = Config(
        task_name="test_task",
        user_id="user1",
        openai={
            "prompt": "input_text",
            "system_instruction": {"is_dynamic": False, "system_prompt": ""},
            "api_key": "dummy",
            "api_information": {"model": "gpt-4", "setting": {"temperature": 0.7, "max_tokens": 10}},
            "batch": {"is_batch_active": False, "batch_size": 1, "batch_check_time": 10, "combine_batches": False, "file_batch": False},
            "bulk_save": 1,
            "sleep_time": 0,
            "proxies": {"proxies_enabled": False, "http": "", "https": ""}
        },
        general=GeneralConfig(
            verbose=True,
            sampling=Sampling(enable_sampling=False, sample_size=1),
            logs=GeneralLoggingConfig(
                log_file="log.log", log_to_file=False, log_to_db=False, log_buffer_size=10
            ),
            multiprocessing_enabled=False,
            num_processes=1,
            db=DBConfig(postgres={}),
            similarity_check=SimilarityCheck()
        )
    )
    object.__setattr__(cfg, "workdir", str(tmp_path))
    return cfg

@pytest.mark.asyncio
async def test_execute_single(tmp_path, dummy_config):
    api = DummyAPI()
    cache = DummyCache()
    service = RequestService(api, cache, dummy_config)
    # Prepare input files
    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    outputs_dir.mkdir()
    (inputs_dir / "input_0.json").write_text('[{"input_text": "hello"}]')
    task = type("Task", (), {"status": None, "task_name": "test_task"})()
    result = await service.execute(task, tmp_path)
    print("Task status:", task.status, "Result:", result)
    print("Inputs dir:", list((tmp_path / "inputs").glob("*")))
    print("Outputs dir:", list((tmp_path / "outputs").glob("*")) if (tmp_path / "outputs").exists() else "No outputs")
    assert task.status.value == "success"

@pytest.mark.asyncio
async def test_execute_batch(tmp_path, dummy_config):
    api = DummyAPI()
    cache = DummyCache()
    service = RequestService(api, cache, dummy_config)
    # Prepare input files
    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    outputs_dir.mkdir()
    (inputs_dir / "input_0.json").write_text('[{"input_text": "hello"}]')
    task = type("Task", (), {"status": None, "task_name": "test_task"})()
    result = await service.execute_batch(task, tmp_path, batch_size=1)
    print("Task status:", task.status, "Result:", result)
    print("Inputs dir:", list((tmp_path / "inputs").glob("*")))
    print("Outputs dir:", list((tmp_path / "outputs").glob("*")) if (tmp_path / "outputs").exists() else "No outputs")
    assert task.status.value == "success"

def test_create_and_list_task(temp_task_dir, monkeypatch):
    monkeypatch.setattr("mitrailleuse.application.services.task_service.TASK_ROOT", temp_task_dir)
    cfg = minimal_config()
    task = TaskService.create_task("user1", "openai", "test_task", cfg)
    assert task.task_name == "test_task"
    tasks = TaskService.list_available_tasks("user1", "test_task")
    print("Tasks found:", tasks)
    assert any(t.task_name == "test_task" for t in tasks)