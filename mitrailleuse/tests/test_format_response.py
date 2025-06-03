import json
from mitrailleuse.scripts.format_response import ResponseFormatter
from pathlib import Path

def test_format_batch_response(tmp_path):
    # Prepare dummy config and input/output
    task_dir = tmp_path / "user1" / "test_task_20240101_000000"
    (task_dir / "inputs").mkdir(parents=True)
    (task_dir / "outputs").mkdir(parents=True)
    (task_dir / "config").mkdir(parents=True)
    config = {
        "openai": {
            "prompt": "input_text",
            "system_instruction": {"is_dynamic": False, "system_prompt": ""},
        }
    }
    with open(task_dir / "config" / "config.json", "w") as f:
        json.dump(config, f)
    with open(task_dir / "inputs" / "input_0.jsonl", "w") as f:
        f.write(json.dumps({"input_text": "hi"}) + "\n")
    with open(task_dir / "outputs" / "input_0_raw_response.json", "w") as f:
        json.dump([{"choices": [{"message": {"content": "hello"}}]}], f)
    formatter = ResponseFormatter("user1", "test_task_20240101_000000", task_dir)
    formatted = formatter.format_batch_response(task_dir / "outputs" / "input_0_raw_response.json")
    assert isinstance(formatted, list)
