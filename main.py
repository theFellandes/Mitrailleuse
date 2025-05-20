"""
Mitrailleuse CLI - A command-line interface for the Mitrailleuse gRPC service

Requirements:
pip install pydantic grpcio grpcio-tools httpx
(plus openai key in the config.json)
"""
import json
import time
import grpc
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.config.config import Config
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter

class MitrailleuseCLI:
    def __init__(self, host: str = "localhost", port: int = 50051):
        """Initialize the CLI with gRPC connection."""
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = mitrailleuse_pb2_grpc.MitrailleuseServiceStub(self.channel)
        self.base_config = json.loads(Path("mitrailleuse/config/config.json").read_text())

    def _s(self, obj) -> str:
        """Cast Path to str, leave str unchanged."""
        return str(obj) if isinstance(obj, Path) else obj

    def create_task(self, user_id: str, api_name: str, task_name: str, config_json: Optional[str] = None) -> Path:
        """Create a new task."""
        if config_json is None:
            config_json = json.dumps(self.base_config, ensure_ascii=False)

        response = self.stub.CreateTask(
            mitrailleuse_pb2.CreateTaskRequest(
                user_id=user_id,
                api_name=api_name,
                task_name=task_name,
                config_json=config_json
            )
        )
        task_dir = Path(response.task_folder)
        print(f"✅ Task created at: {task_dir}")
        return task_dir

    def list_tasks(self, user_id: str, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tasks for a user."""
        response = self.stub.ListTasks(
            mitrailleuse_pb2.ListTasksRequest(
                user_id=user_id,
                task_name=task_name or ""
            )
        )
        
        tasks = []
        if not response.tasks:
            print("No tasks found.")
            return tasks

        print("\nAvailable Tasks:")
        print("-" * 80)
        for i, task in enumerate(response.tasks, 1):
            print(f"{i}. Task Name: {task.task_name}")
            print(f"   API: {task.api_name}")
            print(f"   Status: {task.status}")
            print(f"   Path: {task.path}")
            print("-" * 80)
            tasks.append({
                "index": i,
                "task_name": task.task_name,
                "api_name": task.api_name,
                "status": task.status,
                "path": task.path
            })
        return tasks

    def select_task(self, user_id: str, task_name: Optional[str] = None) -> Optional[str]:
        """Select a task interactively."""
        tasks = self.list_tasks(user_id, task_name)
        if not tasks:
            return None

        while True:
            try:
                choice = int(input("\nSelect a task number (or 0 to cancel): "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(tasks):
                    return tasks[choice - 1]["path"]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def get_task_by_path(self, task_path: str) -> None:
        """Get task information from a specific path."""
        response = self.stub.GetTaskByPath(
            mitrailleuse_pb2.GetTaskByPathRequest(
                task_path=task_path
            )
        )
        
        if not response.user_id:  # Empty response indicates task not found
            print("Task not found.")
            return

        print("\nTask Information:")
        print("-" * 80)
        print(f"User ID: {response.user_id}")
        print(f"Task Name: {response.task_name}")
        print(f"API: {response.api_name}")
        print(f"Status: {response.status}")
        print(f"Path: {response.path}")
        print("-" * 80)

    def execute_task(self, user_id: str, task_folder: Optional[str] = None, task_name: Optional[str] = None) -> None:
        """Execute a task."""
        if task_folder is None:
            task_folder = self.select_task(user_id, task_name)
            if task_folder is None:
                print("No task selected.")
                return

        response = self.stub.ExecuteTask(
            mitrailleuse_pb2.ExecuteTaskRequest(
                user_id=user_id,
                task_folder=task_folder
            )
        )

        if response.status == "failed":
            print("🚨 Task failed – see log:", Path(task_folder) / "logs" / "log.log")
        elif response.job_id:
            print("📡 Batch launched – ID:", response.job_id)
            self._monitor_batch_job(response.job_id, task_folder)
        else:
            print("✅ Single-request task finished")

    def _monitor_batch_job(self, job_id: str, task_folder: str) -> None:
        """Monitor a batch job's status."""
        adapter = OpenAIAdapter(self.base_config)
        print("\nMonitoring batch job status...")
        
        while True:
            status = adapter.get_batch_status(job_id)
            print(f"📡 Batch state: {status['status']}")
            
            if status["status"] in {"completed", "failed"}:
                break
                
            time.sleep(5)

        print("🎉 Done – results are in", Path(task_folder) / "outputs")

def main():
    parser = argparse.ArgumentParser(description="Mitrailleuse CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create task command
    create_parser = subparsers.add_parser("create", help="Create a new task")
    create_parser.add_argument("--user-id", required=True, help="User ID")
    create_parser.add_argument("--api-name", required=True, help="API name (e.g., openai, deepseek)")
    create_parser.add_argument("--task-name", required=True, help="Task name")
    create_parser.add_argument("--config", help="Path to custom config file")

    # List tasks command
    list_parser = subparsers.add_parser("list", help="List available tasks")
    list_parser.add_argument("--user-id", required=True, help="User ID")
    list_parser.add_argument("--task-name", help="Filter by task name")

    # Get task command
    get_parser = subparsers.add_parser("get", help="Get task information")
    get_parser.add_argument("--task-path", required=True, help="Path to task directory")

    # Execute task command
    execute_parser = subparsers.add_parser("execute", help="Execute a task")
    execute_parser.add_argument("--user-id", required=True, help="User ID")
    execute_parser.add_argument("--task-folder", help="Path to task directory")
    execute_parser.add_argument("--task-name", help="Task name to select from")

    args = parser.parse_args()
    cli = MitrailleuseCLI()

    try:
        if args.command == "create":
            config_json = None
            if args.config:
                config_json = Path(args.config).read_text()
            cli.create_task(args.user_id, args.api_name, args.task_name, config_json)
        
        elif args.command == "list":
            cli.list_tasks(args.user_id, args.task_name)
        
        elif args.command == "get":
            cli.get_task_by_path(args.task_path)
        
        elif args.command == "execute":
            cli.execute_task(args.user_id, args.task_folder, args.task_name)
        
        else:
            parser.print_help()

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.details()}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
