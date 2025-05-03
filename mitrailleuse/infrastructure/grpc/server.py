import json
import grpc
from concurrent import futures
from pathlib import Path
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.application.services.task_service import TaskService
from mitrailleuse.application.services.request_service import RequestService, ADAPTERS
from mitrailleuse.infrastructure.adapters.file_cache_adapter import FileCache
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.settings import TASK_ROOT
from mitrailleuse.config.config import Config

ROOT = Path.cwd() / "tasks"
log = get_logger(__name__)


class MitrailleuseGRPC(mitrailleuse_pb2_grpc.MitrailleuseServiceServicer):

    def CreateTask(self, request, context):
        try:
            # Postman can send either a raw JSON string or a Struct‑like object
            cfg_dict = (json.loads(request.config_json)
                        if isinstance(request.config_json, str)
                        else json.loads(json.dumps(request.config_json)))
            cfg = Config.model_validate(cfg_dict)

            task = TaskService.create_task(
                request.user_id,
                request.api_name.lower(),  # ← persist provider name
                request.task_name,
                cfg
            )
            return mitrailleuse_pb2.CreateTaskResponse(
                task_folder=str(task.path(TASK_ROOT))
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
        except Exception as exc:
            log.error("Task creation failed: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
        # ALWAYS return the right message type
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
            task = TaskService.status_from_path(base_path)

            # 2️⃣  choose adapter
            provider = (task.api_name or "openai").lower()
            adapter = ADAPTERS.get(provider, OpenAIAdapter)
            api = adapter(cfg)

            # 3️⃣  fire the RequestService
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
    # ---- Health service ----
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    # Mark all services as SERVING
    health_servicer.set('', health_pb2.HealthCheckResponse.SERVING)

    # ---- Reflection ----
    SERVICE_NAMES = (
        mitrailleuse_pb2.DESCRIPTOR.services_by_name['MitrailleuseService'].full_name,
        health.SERVICE_NAME,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port("[::]:50051")
    log.info("Server started on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    log.info("Server starting...")
    serve()
    log.info("Server stopped")
