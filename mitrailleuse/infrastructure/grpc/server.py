import json
import grpc
from concurrent import futures
from pathlib import Path
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
import asyncio
import logging
from datetime import datetime

from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.application.services.task_service import TaskService
from mitrailleuse.application.services.request_service import RequestService, ADAPTERS
from mitrailleuse.infrastructure.adapters.file_cache_adapter import FileCache
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
from mitrailleuse.infrastructure.logging.logger import get_logger
from mitrailleuse.infrastructure.settings import TASK_ROOT
from mitrailleuse.config.config import Config

ROOT = TASK_ROOT

# Set up server logging
server_log_dir = Path(__file__).parent.parent.parent.parent / "logs"
server_log_dir.mkdir(parents=True, exist_ok=True)
server_log_file = server_log_dir / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure server logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(server_log_file),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


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
        try:
            # Get all tasks for the user
            user_root = ROOT / request.user_id
            if not user_root.exists():
                err_msg = f"No tasks found for user {request.user_id}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Collect all task directories
            task_dirs = sorted([d for d in user_root.iterdir() if d.is_dir()], key=lambda x: x.name)
            
            # If task_folder is a number, treat it as an index
            if request.task_folder and request.task_folder.isdigit():
                try:
                    task_idx = int(request.task_folder) - 1  # Convert to 0-based index
                    if 0 <= task_idx < len(task_dirs):
                        task_path = task_dirs[task_idx]
                        log.info(f"Selected task {task_idx + 1}: {task_path}")
                    else:
                        err_msg = f"Invalid task number: {request.task_folder}. Available tasks: 1-{len(task_dirs)}"
                        log.error(err_msg)
                        context.set_details(err_msg)
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")
                except ValueError:
                    err_msg = f"Invalid task number format: {request.task_folder}"
                    log.error(err_msg)
                    context.set_details(err_msg)
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")
            else:
                task_path = Path(request.task_folder)

            if not task_path.exists():
                err_msg = f"Task folder {task_path} does not exist"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Load configuration
            config_path = task_path / "config" / "config.json"
            if not config_path.exists():
                err_msg = f"Config file not found at {config_path}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Read and validate config
            try:
                cfg = Config.read(config_path)
            except Exception as e:
                err_msg = f"Failed to load config: {str(e)}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Get task status
            try:
                task = TaskService.status_from_path(task_path)
            except Exception as e:
                err_msg = f"Failed to get task status: {str(e)}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Initialize cache
            try:
                cache = FileCache(task_path / "cache")
            except Exception as e:
                err_msg = f"Failed to initialize cache: {str(e)}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Choose adapter
            try:
                provider = (task.api_name or "openai").lower()
                adapter = ADAPTERS.get(provider, OpenAIAdapter)
                # Convert Config to dictionary before passing to adapter
                config_dict = cfg.model_dump()
                api = adapter(config_dict)
            except Exception as e:
                err_msg = f"Failed to initialize API adapter: {str(e)}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

            # Fire the RequestService
            try:
                # Create an event loop and run the async execute method
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                job_id = loop.run_until_complete(RequestService(api, cache, cfg).execute(task, task_path))
                loop.close()
            except Exception as e:
                err_msg = f"Failed to execute task: {str(e)}"
                log.error(err_msg)
                context.set_details(err_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                return mitrailleuse_pb2.ExecuteTaskResponse(status="failed", job_id="")

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

    def ListTasks(self, request, context):
        try:
            tasks = []
            user_root = ROOT / request.user_id
            if not user_root.exists():
                log.info(f"No tasks directory found for user {request.user_id}")
                return mitrailleuse_pb2.ListTasksResponse(tasks=[])

            # Get all task directories and sort them
            task_dirs = sorted([d for d in user_root.iterdir() if d.is_dir()], key=lambda x: x.name)
            if not task_dirs:
                log.info(f"No tasks found in directory {user_root}")
                return mitrailleuse_pb2.ListTasksResponse(tasks=[])

            log.info(f"Found {len(task_dirs)} tasks for user {request.user_id}")
            
            # Create task info for each directory
            for task_dir in task_dirs:
                try:
                    task = TaskService.status_from_path(task_dir)
                    tasks.append(mitrailleuse_pb2.TaskInfo(
                        user_id=request.user_id,
                        api_name=task.api_name,
                        task_name=task.task_name,
                        status=task.status.value,
                        path=str(task_dir)
                    ))
                except Exception as e:
                    log.error(f"Error processing task {task_dir}: {str(e)}")
                    continue

            return mitrailleuse_pb2.ListTasksResponse(tasks=tasks)
        except Exception as exc:
            log.error(f"List tasks failed: {str(exc)}")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return mitrailleuse_pb2.ListTasksResponse(tasks=[])

    def GetTaskByPath(self, request, context):
        try:
            task_path = Path(request.task_path)
            if not task_path.exists():
                context.set_details(f"Task folder {task_path} does not exist")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return mitrailleuse_pb2.TaskInfo()

            task = TaskService.status_from_path(task_path)
            return mitrailleuse_pb2.TaskInfo(
                user_id=task.user_id,
                api_name=task.api_name,
                task_name=task.task_name,
                status=task.status.value,
                path=str(task_path)
            )
        except Exception as exc:
            log.error(f"Get task by path failed: {str(exc)}")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return mitrailleuse_pb2.TaskInfo()


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
