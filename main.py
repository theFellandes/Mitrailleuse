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
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from mitrailleuse import mitrailleuse_pb2, mitrailleuse_pb2_grpc
from mitrailleuse.config.config import Config
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
import asyncio

# Define the root directory for task outputs
ROOT = Path("tasks")

# Set up logging directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / f'mitrailleuse_cli_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class MitrailleuseCLI:
    def __init__(self, host: str = "localhost", port: int = 50051):
        """Initialize the CLI with gRPC connection."""
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = mitrailleuse_pb2_grpc.MitrailleuseServiceStub(self.channel)
        self.base_config = json.loads(Path("mitrailleuse/config/config.json").read_text())
        logger.info(f"Initialized MitrailleuseCLI with connection to {host}:{port}")

    def _s(self, obj) -> str:
        """Cast Path to str, leave str unchanged."""
        return str(obj) if isinstance(obj, Path) else obj

    def create_task(self, user_id: str, api_name: str, task_name: str, config_json: Optional[str] = None) -> Path:
        """Create a new task."""
        logger.info(f"Creating new task - User: {user_id}, API: {api_name}, Task: {task_name}")
        start_time = time.time()
        
        if config_json is None:
            config_json = json.dumps(self.base_config, ensure_ascii=False)
            logger.debug("Using default configuration")

        try:
            response = self.stub.CreateTask(
                mitrailleuse_pb2.CreateTaskRequest(
                    user_id=user_id,
                    api_name=api_name,
                    task_name=task_name,
                    config_json=config_json if config_json is not None else ""
                )
            )
            task_dir = Path(response.task_folder)
            duration = time.time() - start_time
            logger.info(f"✅ Task created successfully in {duration:.2f}s at: {task_dir}")
            return task_dir
        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            raise

    def list_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """List available tasks for a user."""
        logger.info(f"Listing tasks for user: {user_id}")
        try:
            response = self.stub.ListTasks(
                mitrailleuse_pb2.ListTasksRequest(
                    user_id=user_id,
                    task_name=""  # Always list all tasks
                )
            )
            
            tasks = []
            if not response.tasks:
                logger.info("No tasks found")
                print("No tasks found.")
                return tasks

            # Group tasks by user_id
            tasks_by_user = {}
            for task in response.tasks:
                if task.user_id not in tasks_by_user:
                    tasks_by_user[task.user_id] = []
                tasks_by_user[task.user_id].append(task)

            logger.info(f"Found tasks for {len(tasks_by_user)} users")
            
            # Print tasks grouped by user
            for user_id, user_tasks in tasks_by_user.items():
                print(f"\nTasks for User: {user_id}")
                print("=" * 80)
                
                for i, task in enumerate(user_tasks, 1):
                    print(f"#{i}")
                    print(f"Task Name: {task.task_name}")
                    print(f"API: {task.api_name}")
                    print(f"Status: {task.status}")
                    print(f"Path: {task.path}")
                    print("-" * 80)
                    
                    tasks.append({
                        "user_id": task.user_id,
                        "task_number": i,
                        "task_name": task.task_name,
                        "api_name": task.api_name,
                        "status": task.status,
                        "path": task.path
                    })
            
            return tasks
        except Exception as e:
            logger.error(f"Failed to list tasks: {str(e)}")
            raise

    def select_task(self, user_id: str, task_name: Optional[str] = None) -> Optional[str]:
        """Select a task interactively."""
        tasks = self.list_tasks(user_id)
        if not tasks:
            return None

        while True:
            try:
                choice = int(input("\nSelect a task number (or 0 to cancel): "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(tasks):
                    return tasks[choice - 1]["task_id"]
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

    def execute_task(self, user_id: str, task_name: Optional[str] = None, task_number: Optional[str] = None) -> None:
        """Execute a task."""
        start_time = time.time()

        # Prepare variables for logging
        selected_task_name = task_name
        selected_task_number = task_number
        task_path = None

        # If task_number is provided, fetch the task info for logging
        if task_number:
            # Get the list of tasks for the user
            response = self.stub.ListTasks(
                mitrailleuse_pb2.ListTasksRequest(
                    user_id=user_id,
                    task_name=""
                )
            )
            tasks = [t for t in response.tasks]
            try:
                idx = int(task_number) - 1
                if 0 <= idx < len(tasks):
                    task_path = Path(tasks[idx].path)
                else:
                    logger.error(f"Invalid task number: {task_number}")
                    print(f"❌ Invalid task number: {task_number}")
                    return
            except Exception as e:
                logger.error(f"Error resolving task number: {str(e)}")
                print(f"❌ Error resolving task number: {task_number}")
                return
        else:
            task_path = Path(ROOT) / user_id / task_name

        # Always read config to get the canonical task name and is_batch
        config_path = task_path / "config" / "config.json"
        is_batch = False
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                selected_task_name = config.get("task_name", selected_task_name)
                is_batch = config.get("openai", {}).get("batch", {}).get("is_batch_active", False)
        logger.info(f"[Task Info] name: {selected_task_name}, number: {selected_task_number}, is_batch: {is_batch}")

        try:
            logger.info(f"Sending execute request for task name: {selected_task_name}, number: {selected_task_number}")
            response = self.stub.ExecuteTask(
                mitrailleuse_pb2.ExecuteTaskRequest(
                    user_id=user_id,
                    task_folder=task_number or task_name  # Use task number or name directly
                )
            )

            if response.status == "failed":
                duration = time.time() - start_time
                logger.error(f"Task failed after {duration:.2f}s [name: {selected_task_name}, number: {selected_task_number}]")
                print(f"🚨 Task failed – see log: {Path(ROOT) / user_id / 'logs' / 'log.log'}")
            elif response.job_id:
                logger.info(f"Batch job launched with ID: {response.job_id} [name: {selected_task_name}, number: {selected_task_number}]")
                print(f"📡 Batch launched – ID: {response.job_id} (task name: {selected_task_name}, number: {selected_task_number})")
                self._monitor_batch_job(response.job_id, task_path)
            else:
                duration = time.time() - start_time
                # Re-read config in case it was updated during execution
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        is_batch = config.get("openai", {}).get("batch", {}).get("is_batch_active", False)
                logger.info(f"[Post-exec] is_batch: {is_batch} for task name: {selected_task_name}, number: {selected_task_number}")
                if is_batch:
                    logger.info(f"Batch task completed in {duration:.2f}s [name: {selected_task_name}, number: {selected_task_number}]")
                    print(f"✅ Batch task finished (task name: {selected_task_name}, number: {selected_task_number})")
                else:
                    logger.info(f"Single-request task completed in {duration:.2f}s [name: {selected_task_name}, number: {selected_task_number}]")
                    print(f"✅ Single-request task finished (task name: {selected_task_name}, number: {selected_task_number})")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task execution failed after {duration:.2f}s: {str(e)}")
            raise

    async def _monitor_batch_job(self, job_id: str, task_folder: str) -> None:
        """Monitor a batch job's status."""
        logger.info(f"Starting batch job monitoring - Job ID: {job_id}")
        
        # Load task config
        config_path = Path(task_folder) / "config" / "config.json"
        if not config_path.exists():
            logger.error(f"Config file not found at {config_path}")
            return
            
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            
        adapter = OpenAIAdapter(cfg)
        print("\nMonitoring batch job status...")
        
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                status = await adapter.get_batch_status(job_id)
                current_status = status['status']
                
                # Only log status changes
                if current_status != last_status:
                    logger.info(f"Batch job status changed to: {current_status}")
                    last_status = current_status
                
                # Show progress if available
                if 'completed_count' in status and 'total_count' in status:
                    progress = (status['completed_count'] / status['total_count']) * 100
                    print(f"📡 Batch state: {current_status} ({progress:.1f}% complete)")
                else:
                    print(f"📡 Batch state: {current_status}")
                
                if current_status in {"completed", "failed", "expired", "cancelled"}:
                    duration = time.time() - start_time
                    if current_status == "completed":
                        logger.info(f"Batch job completed successfully in {duration:.2f}s")
                        # Download results
                        try:
                            output_path = await adapter.download_batch_results(job_id, Path(task_folder) / "outputs", "batch")
                            logger.info(f"Results downloaded to: {output_path}")
                            print(f"✅ Results saved to: {output_path}")
                        except Exception as e:
                            logger.error(f"Error downloading results: {str(e)}")
                            print(f"⚠️ Error downloading results: {str(e)}")
                    else:
                        logger.error(f"Batch job failed after {duration:.2f}s with status: {current_status}")
                        print(f"❌ Batch job failed with status: {current_status}")
                    break
                    
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error monitoring batch job: {str(e)}")
                raise

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
    list_parser.add_argument("--user-id", required=True, help="User ID (use 'all' to list tasks for all users)")
    list_parser.add_argument("--all", action="store_true", help="List tasks for all users (alternative to --user-id all)")

    # Get task command
    get_parser = subparsers.add_parser("get", help="Get task information")
    get_parser.add_argument("--task-path", required=True, help="Path to task directory")

    # Execute task command
    execute_parser = subparsers.add_parser("execute", help="Execute a task")
    execute_parser.add_argument("--user-id", required=True, help="User ID")
    execute_parser.add_argument("--task-name", help="Task name to execute")
    execute_parser.add_argument("--task-number", help="Task number to execute")

    args = parser.parse_args()
    cli = MitrailleuseCLI()

    try:
        if args.command == "create":
            logger.info("Starting task creation")
            config_json = None
            if args.config:
                config_json = Path(args.config).read_text()
                logger.info(f"Using custom config from: {args.config}")
            cli.create_task(args.user_id, args.api_name, args.task_name, config_json)
        
        elif args.command == "list":
            logger.info("Starting task listing")
            # Use 'all' if either --all flag is set or user_id is 'all'
            user_id = "all" if args.all or args.user_id.lower() == "all" else args.user_id
            cli.list_tasks(user_id)
        
        elif args.command == "get":
            logger.info(f"Getting task information for: {args.task_path}")
            cli.get_task_by_path(args.task_path)
        
        elif args.command == "execute":
            logger.info("Starting task execution")
            cli.execute_task(args.user_id, args.task_name, args.task_number)
        
        else:
            parser.print_help()

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.details()}")
        print(f"gRPC Error: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
