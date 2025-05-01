"""
requirements:
pip install pydantic grpcio grpcio-tools httpx
(plus openai key in the JSON above)
"""
import json, time, grpc, copy
from pathlib import Path
from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.config.config import Config
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter

channel   = grpc.insecure_channel("localhost:50051")
stub      = mitrailleuse_pb2_grpc.MitrailleuseServiceStub(channel)

# ------------------------------------------------------------------ 1) create task
base_cfg  = json.loads(Path("mitrailleuse/config/config.json").read_text())

create_rsp = stub.CreateTask(
    mitrailleuse_pb2.CreateTaskRequest(
        user_id   = "owx123456",
        api_name  = "openai",
        task_name = "few_shot_demo",
        config_json = json.dumps(base_cfg, ensure_ascii=False)   # pristine on first call
    )
)
task_dir = Path(create_rsp.task_folder)
print("✅ task created at", task_dir)

# ------------------------------------------------------------------ 2) dynamic override
override_cfg          = copy.deepcopy(base_cfg)
override_cfg["general"]["multiprocessing_enabled"] = False    # single-proc
override_cfg["openai"]["batch"]["is_batch_active"] = True     # force batch

# we *re-use* the same task folder, so server will just update config.json
stub.ExecuteTask.future  # we’ll call it later, after dropping inputs

(task_dir / "config" / "config.json").write_text(json.dumps(override_cfg, indent=2))
print("🔁  wrote overridden config into", task_dir / "config" / "config.json")

# ------------------------------------------------------------------ drop a couple of JSON inputs
sample_req = {
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": "Say hello from the Mitrailleuse demo"}
    ]
}
for i in range(3):
    (task_dir / "inputs" / f"input_{i}.json").write_text(json.dumps(sample_req))

# ------------------------------------------------------------------ 3) run batch request
exec_rsp = stub.ExecuteTask(
    mitrailleuse_pb2.ExecuteTaskRequest(
        user_id     = "alice",
        task_folder = str(task_dir)
    )
)
print("⚙️  execution kicked off – server returned status:", exec_rsp.status)

# grab the persisted batch id
batch_job_file = task_dir / "cache" / "batch_job.json"
job_id         = json.loads(batch_job_file.read_text())["id"]

# ------------------------------------------------------------------ 4) poll batch status
adapter = OpenAIAdapter(Config(**override_cfg))
while True:
    status = adapter.get_batch_status(job_id)
    print("📡  batch state:", status["status"])
    if status["status"] in {"completed", "failed"}:
        break
    time.sleep(5)

print("🎉  done – results are in", task_dir / "outputs")
