import json
import grpc
from concurrent import futures
from pathlib import Path

from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.application.services.task_service import TaskService
from mitrailleuse.application.services.request_service import RequestService
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
from mitrailleuse.infrastructure.adapters.file_cache_adapter import FileCache
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.settings import TASK_ROOT
from mitrailleuse.config.config import Config

ROOT = Path.cwd() / "tasks"
log = get_logger(__name__)


class MitrailleuseGRPC(mitrailleuse_pb2_grpc.MitrailleuseServiceServicer):

    def CreateTask(self, request, context):
        try:
            cfg = Config.model_validate(json.loads(request.config_json))
            task = TaskService.create_task(request.user_id, request.api_name,
                                         request.task_name, cfg)
            return mitrailleuse_pb2.CreateTaskResponse(
                task_folder=str(task.path(TASK_ROOT))
            )
        except Exception as exc:
            log.error(f"Task creation failed: {str(exc)}")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return mitrailleuse_pb2.CreateTaskResponse(task_folder="")

    def ExecuteTask(self, request, context):
        base_path = Path(request.task_folder)

        try:
            # Verify the task folder exists
            if not base_path.exists():
                err_msg = f"Task folder {base_path} does not exist"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Load configuration
            config_path = base_path / "config" / "config.json"
            if not config_path.exists():
                err_msg = f"Config file not found at {config_path}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            cfg = Config.read(config_path)
            cache = FileCache(base_path / "cache")
            api = OpenAIAdapter(cfg)
            task = TaskService.status_from_path(base_path)

            # Create services and execute
            job_id = RequestService(api, cache, cfg).execute(task, base_path)
            return mitrailleuse_pb2.ExecuteTaskResponse(
                status=task.status.value,
                job_id=job_id or ""
            )

        except Exception as exc:
            log.error(f"Task execution failed: {str(exc)}")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

    def GetTaskStatus(self, request, context):
        try:
            task_path = Path(request.task_folder)
            if not task_path.exists():
                context.set_details(f"Task folder {task_path} does not exist")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.GetTaskStatusResponse(status="unknown")

            task = TaskService.status_from_path(task_path)
            return mitrailleuse_pb2.GetTaskStatusResponse(status=task.status.value)
        except Exception as exc:
            log.error(f"Status check failed: {str(exc)}")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return mitrailleuse_pb2.GetTaskStatusResponse(status="unknown")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mitrailleuse_pb2_grpc.add_MitrailleuseServiceServicer_to_server(
        MitrailleuseGRPC(), server)
    server.add_insecure_port("[::]:50051")
    log.info("Server started on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    log.info("Server starting...")
    serve()
    log.info("Server stopped")
