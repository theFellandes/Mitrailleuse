import json
import grpc
from concurrent import futures
from pathlib import Path

from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.application.services.task_service import TaskService
from mitrailleuse.application.services.request_service import RequestService
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
from mitrailleuse.infrastructure.adapters.file_cache_adapter import FileCache
from mitrailleuse.infrastructure.settings import TASK_ROOT
from mitrailleuse.config.config import Config

ROOT = Path.cwd() / "tasks"


class MitrailleuseGRPC(mitrailleuse_pb2_grpc.MitrailleuseServiceServicer):

    def CreateTask(self, request, context):
        cfg = Config(**json.loads(request.config_json))
        task = TaskService.create_task(request.user_id, request.api_name,
                                       request.task_name, cfg)
        return mitrailleuse_pb2.CreateTaskResponse(
            task_folder=str(task.path(TASK_ROOT))  # <-- use the canonical root
        )

    def ExecuteTask(self, request, context):
        task_path = Path(request.task_folder)
        cfg = Config.read(task_path / "config" / "config.json")
        cache = FileCache(task_path / "cache")
        api = OpenAIAdapter(cfg)
        task = TaskService.status_from_path(Path(request.task_folder))  # helper you can add
        RequestService(api, cache, cfg).execute(task)
        return mitrailleuse_pb2.ExecuteTaskResponse(status=task.status)

    def GetTaskStatus(self, request, context):
        task_path = Path(request.task_folder)
        task = TaskService.status_from_path(task_path)
        return mitrailleuse_pb2.GetTaskStatusResponse(status=task.status)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mitrailleuse_pb2_grpc.add_MitrailleuseServiceServicer_to_server(
        MitrailleuseGRPC(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    print("Server Started")
    serve()
    print("Server Dieded")
