import grpc
from pathlib import Path
from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.config.config import Config

channel = grpc.insecure_channel("localhost:50051")
stub = mitrailleuse_pb2_grpc.MitrailleuseServiceStub(channel)

# ---- 1. Create a task -------------------------------------------------------
config = Config.read(Path("sample_config.json"))  # your existing file
create_resp = stub.CreateTask(
    mitrailleuse_pb2.CreateTaskRequest(
        user_id="alice",
        api_name="openai",
        task_name="few_shot_test",
        config_json=config.model_dump_json()
    )
)
print("Task folder:", create_resp.task_folder)

# ---- 2. Drop your JSON inputs in {task_folder}/inputs/ manually ------------
# ---- 3. Execute -------------------------------------------------------------
exec_resp = stub.ExecuteTask(
    mitrailleuse_pb2.ExecuteTaskRequest(
        user_id="alice", task_folder=create_resp.task_folder
    )
)
print("Execution status:", exec_resp.status)

# ---- 4. Query status any time ----------------------------------------------
status_resp = stub.GetTaskStatus(
    mitrailleuse_pb2.GetTaskStatusRequest(
        user_id="alice", task_folder=create_resp.task_folder
    )
)
print("Current status:", status_resp.status)
