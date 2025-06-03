import types
import pytest
from mitrailleuse.application.services.task_service import TaskService
from mitrailleuse.config.config import Config, GeneralConfig, Sampling, GeneralLoggingConfig, DBConfig, SimilarityCheck
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter

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

def test_create_and_list_task(temp_task_dir, monkeypatch):
    monkeypatch.setattr("mitrailleuse.application.services.task_service.TASK_ROOT", temp_task_dir)
    cfg = minimal_config()
    task = TaskService.create_task("user1", "openai", "test_task", cfg)
    assert task.task_name == "test_task"
    tasks = TaskService.list_available_tasks("user1", "test_task")
    print("Tasks found:", tasks)
    assert any(t.task_name == "test_task" for t in tasks)


@pytest.mark.asyncio
async def test_openai_adapter_ping(monkeypatch):
    config = {"openai": {"api_key": "sk-xxx", "api_information": {"model": "gpt-4", "setting": {"temperature": 0.7, "max_tokens": 10}}}}
    adapter = OpenAIAdapter(config)
    async def dummy_create(*args, **kwargs):
        return {'choices':[{'message':{'content':'ok'}}]}
    dummy_completions = type("Completions", (), {"create": dummy_create})()
    dummy_chat = type("Chat", (), {"completions": dummy_completions})()
    dummy_client = type("Dummy", (), {"chat": dummy_chat})()
    monkeypatch.setattr(adapter, "client", dummy_client)
    result = await adapter.send_single({"messages": [{"role": "user", "content": "hi"}]})
    assert "choices" in result or "content" in result