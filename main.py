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


# 2) prep helper + values ---------------------------------------------------
def _s(obj):           # cast Path → str, leave str unchanged
    return str(obj) if isinstance(obj, Path) else obj


user_id   = "owx123456"
task_dir  = Path(
    r"D:\Programming\Python\Mitrailleuse\mitrailleuse\tasks"
    r"\owx123456\openai_few_shot_demo_01_05_2025_131432"
)

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
override_cfg["openai"]["batch"]["is_batch_active"] = False     # force batch

# we *re-use* the same task folder, so server will just update config.json
stub.ExecuteTask.future  # we’ll call it later, after dropping inputs

(task_dir / "config" / "config.json").write_text(json.dumps(override_cfg, indent=2))
print("🔁  wrote overridden config into", task_dir / "config" / "config.json")

# ------------------------------------------------------------------ drop a couple of JSON inputs
sample_req = {
    "instruction": "You are a helpful assistant",
    "user_prompt": "Say hello from the Mitrailleuse demo"
}
for i in range(3):
    (task_dir / "inputs" / f"input_{i}.json").write_text(json.dumps(sample_req))

# 3) **THIS is the block you asked about** ----------------------------------
exec_rsp = stub.ExecuteTask(
    mitrailleuse_pb2.ExecuteTaskRequest(
        user_id    = str(user_id),            # **always cast to str**
        task_folder= str(task_dir)            # idem – Path → str
    )
)

# 4) handle the response ----------------------------------------------------
if exec_rsp.status == "failed":
    print("🚨 task failed – see log:",
          task_dir / "logs" / "log.log")
elif exec_rsp.job_id:
    print("📡 batch launched – id:", exec_rsp.job_id)
else:
    print("✅ single-request task finished")

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
