import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from openai import AsyncOpenAI
import httpx
from datetime import datetime
import logging
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import deepl
from mitrailleuse.infrastructure.utils.file_converter import FileConverter
from mitrailleuse.infrastructure.utils.cache_manager import CacheManager
from mitrailleuse.infrastructure.utils.file_flattener import FileFlattener
from mitrailleuse.infrastructure.utils.logger import get_logger
from mitrailleuse.infrastructure.utils.similarity_checker import SimilarityChecker
from mitrailleuse.infrastructure.utils.prompt_utils import (
    build_openai_body,
    build_deepl_body,
    find_prompt_content
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleClient:
    def __init__(self, config_path: Union[str, Path], task_path: Optional[Union[str, Path]] = None):
        """Initialize the client with configuration."""
        self.config_path = Path(config_path)
        self.task_path = Path(task_path) if task_path else None
        
        # Load initial config to get user_id and task_name
        initial_config = self._load_config(self.config_path)
        self.user_id = initial_config.get("user_id", "default_user")
        self.task_name = initial_config["task_name"]
        
        # Set up task directory
        if self.task_path:
            if not self.task_path.exists():
                raise ValueError(f"Specified task path does not exist: {task_path}")
            # Use task-specific config
            self.base_path = self.task_path
            self.config = self._load_config(self.base_path / "config" / "config.json")
        else:
            self.base_path = self._create_task_directory()
            self.config = self._load_config(self.base_path / "config" / "config.json")
        
        # Initialize API clients
        self.openai_client = None
        self.deepl_client = None
        
        if "openai" in self.config:
            self.openai_client = AsyncOpenAI(api_key=self.config["openai"]["api_key"])
            self.openai_batch_size = self.config["openai"]["batch"]["batch_size"]
            self.openai_is_batch_active = self.config["openai"]["batch"]["is_batch_active"]
            self.openai_combine_batches = self.config.get("openai", {}).get("combine_batches", False)
        
        if "deepl" in self.config:
            self.deepl_client = deepl.Translator(self.config["deepl"]["api_key"])
            self.deepl_batch_size = self.config["deepl"].get("batch_size", 50)
            self.deepl_is_batch_active = self.config["deepl"].get("is_batch_active", True)
            self.deepl_combine_batches = self.config["deepl"].get("combine_batches", True)
        
        # Calculate thread cap based on system resources
        self.max_threads = self._calculate_thread_cap()
        
        self._create_task_directories()
        self._check_input_files()
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.base_path, self.config)
        
        # Initialize file flattener
        self.file_flattener = FileFlattener(self.base_path, self.config)
        
        # Initialize similarity checker if enabled
        self.similarity_checker = None
        if self._check_similarity_enabled():
            self.similarity_checker = SimilarityChecker(self.base_path, self.config)
            logger.info("Similarity checking enabled")

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _create_task_directory(self) -> Path:
        """Create a new task directory with timestamp."""
        script_dir = Path(__file__).parent.absolute()
        tasks_dir = script_dir / "tasks" / self.user_id
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir = tasks_dir / f"{self.task_name}_{timestamp}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the main config to the task's config directory
        config_dir = task_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_dir / "config.json", config_dir / "config.json")
        
        logger.info(f"Created new task directory: {task_dir}")
        return task_dir

    def _create_task_directories(self):
        """Create necessary directories for the task."""
        FileConverter.ensure_directory_structure(self.base_path)

    def _check_input_files(self):
        """Check if input directory has files and prompt user if empty."""
        input_dir = self.base_path / "inputs"
        if not any(input_dir.iterdir()):
            print("\nNo input files found in the task's input directory.")
            print(f"Please add your input files to: {input_dir}")
            print("\nPress Enter when you've added the files, or 'q' to quit...")
            
            while True:
                user_input = input().strip().lower()
                if user_input == 'q':
                    logger.info("User chose to quit. Exiting...")
                    raise SystemExit("No input files provided. Exiting.")
                
                if any(input_dir.iterdir()):
                    logger.info("Input files detected. Proceeding...")
                    break
                else:
                    print("Still no input files found. Press Enter to check again or 'q' to quit...")

    def _list_available_tasks(self) -> List[Path]:
        """List all available tasks for the current user."""
        script_dir = Path(__file__).parent.absolute()
        tasks_dir = script_dir / "tasks" / self.user_id
        if not tasks_dir.exists():
            return []
        
        return sorted(tasks_dir.glob(f"{self.task_name}_*"), reverse=True)

    @staticmethod
    def select_task() -> Optional[Path]:
        """Prompt user to select an existing task or create a new one."""
        script_dir = Path(__file__).parent.absolute()
        config_path = script_dir / "config.json"
        
        # Load config to get user_id and task_name
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        user_id = config.get("user_id", "default_user")
        task_name = config["task_name"]
        
        tasks_dir = script_dir / "tasks" / user_id
        if not tasks_dir.exists():
            return None
        
        available_tasks = sorted(tasks_dir.glob(f"{task_name}_*"), reverse=True)
        if not available_tasks:
            return None
        
        print("\nAvailable tasks:")
        for i, task in enumerate(available_tasks, 1):
            # Try to read task status from status.json
            status = "unknown"
            status_file = task / "status.json"
            if status_file.exists():
                try:
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                        status = status_data.get("status", "unknown")
                except Exception:
                    pass
            
            print(f"{i}. {task.name} (Status: {status})")
        print("0. Create new task")
        
        while True:
            try:
                choice = int(input("\nSelect a task number (or 0 for new): "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(available_tasks):
                    return available_tasks[choice - 1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def _build_request_body(self, input_data: Dict, service: str = "openai") -> Dict:
        """Build the request body for the specified API service."""
        if service == "openai":
            return build_openai_body(input_data, self.config)
        elif service == "deepl":
            return build_deepl_body(input_data, self.config)
        else:
            raise ValueError(f"Unsupported service: {service}")

    def _backup_input_file(self, input_file: Path) -> None:
        """Backup the input file to the backup directory."""
        backup_dir = self.base_path / "inputs" / "backup"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{input_file.stem}_{timestamp}{input_file.suffix}"
        shutil.copy2(input_file, backup_file)
        logger.info(f"Backed up input file to: {backup_file}")

    def _convert_to_jsonl(self, input_file: Path) -> Path:
        """Convert JSON file to JSONL format if needed and backup original."""
        # Backup the original file
        self._backup_input_file(input_file)
        
        if input_file.suffix == '.jsonl':
            return input_file

        output_file = input_file.with_suffix('.jsonl')
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]

        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Converted {input_file} to JSONL format")
        return output_file

    def _convert_to_json(self, input_file: Path) -> Path:
        """Convert JSONL file to JSON format if needed and backup original."""
        # Backup the original file
        self._backup_input_file(input_file)
        
        if input_file.suffix == '.json':
            return input_file

        output_file = input_file.with_suffix('.json')
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Converted {input_file} to JSON format")
        return output_file

    def _split_into_batches(self, input_file: Path, batch_size: int) -> List[Path]:
        """Split input file into batches based on batch size."""
        with open(input_file, 'r') as f:
            items = [json.loads(line) for line in f if line.strip()]

        batch_files = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_file = self.base_path / "inputs" / f"batch_{i//batch_size}.jsonl"
            with open(batch_file, 'w') as f:
                for item in batch:
                    f.write(json.dumps(item) + '\n')
            batch_files.append(batch_file)
        
        return batch_files

    def _calculate_thread_cap(self) -> int:
        """Calculate the maximum number of threads to use."""
        # Get system CPU count
        cpu_count = os.cpu_count() or 4
        
        # Calculate base cap (75% of CPU cores)
        base_cap = max(1, math.floor(cpu_count * 0.75))
        
        # Get config override if exists
        config_cap = self.config.get("general", {}).get("num_processes")
        if config_cap is not None:
            return min(config_cap, base_cap)
        
        return base_cap

    def _check_similarity_enabled(self) -> bool:
        """Check if similarity checking is enabled in config."""
        try:
            return bool(self.config["general"]["check_similarity"])
        except (KeyError, TypeError):
            return False

    def _should_close_similarity_checker(self) -> bool:
        """Check if similarity checker should be closed after each use."""
        try:
            return bool(self.config["general"]["similarity_settings"]["close_after_use"])
        except (KeyError, TypeError):
            return True  # Default to True for safety

    def _should_combine_batches(self, config: Dict) -> bool:
        """Check if batches should be combined based on config."""
        try:
            return bool(config["openai"]["batch"]["combine_batches"])
        except (KeyError, TypeError):
            return False

    def _should_use_file_batch(self, config: Dict) -> bool:
        """Check if file batch mode should be used."""
        try:
            return bool(config["openai"]["batch"]["file_batch"])
        except (KeyError, TypeError):
            return False

    async def _process_single_file(self, input_file: Union[str, Path], service: str, base_path: Union[str, Path], config: Dict) -> Dict:
        """Process a single file and return the result."""
        try:
            # Convert string paths back to Path objects
            input_file = Path(input_file)
            base_path = Path(base_path)
            
            # Initialize components for this process
            cache_manager = CacheManager(base_path, config)
            
            # Backup original file
            original_dir = base_path / "inputs" / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_file, original_dir / input_file.name)
            
            # Convert to JSONL for processing
            jsonl_file = FileConverter.convert_to_jsonl(input_file, base_path)
            
            with open(jsonl_file, 'r') as f:
                input_data = json.loads(f.readline().strip())

            # Check cache first
            cached_response = cache_manager.get(input_data, service)
            if cached_response:
                logger.info(f"Cache hit for {service} request")
                return cached_response

            # Initialize similarity checker only if needed
            similarity_checker = None
            if self._check_similarity_enabled():
                similarity_checker = SimilarityChecker(base_path, config)
                # Check cooldown if similarity checking is enabled
                if similarity_checker.should_cooldown():
                    cooldown_time = similarity_checker.get_cooldown_time()
                    logger.warning(f"Cooldown active. Waiting {cooldown_time} seconds...")
                    await asyncio.sleep(cooldown_time)

            # Build and send request
            request_body = self._build_request_body(input_data, service)
            try:
                if service == "openai":
                    response = await self.openai_client.chat.completions.create(**request_body)
                    # Convert response to dict for JSON serialization
                    result = {
                        "input": input_data,
                        "response": response.model_dump(),
                        "timestamp": datetime.now().isoformat()
                    }
                elif service == "deepl":
                    translation = self.deepl_client.translate_text(
                        request_body["text"],
                        target_lang=request_body["target_lang"],
                        source_lang=request_body["source_lang"]
                    )
                    result = {
                        "input": input_data,
                        "response": {
                            "translated_text": translation.text,
                            "detected_source_lang": translation.detected_source_lang
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Check similarity if enabled
                if similarity_checker:
                    is_similar, similarity = similarity_checker.check_similarity(result["response"])
                    if is_similar:
                        cooldown_time = similarity_checker.get_cooldown_time()
                        logger.warning(
                            f"Similar response detected (similarity: {similarity:.2f}). "
                            f"Applying cooldown of {cooldown_time} seconds."
                        )
                        await asyncio.sleep(cooldown_time)
                
                # Save to cache
                cache_manager.set(input_data, result, service)
                
                # Save both raw and parsed responses
                output_dir = base_path / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save raw response
                output_file = output_dir / f"{input_file.stem}_{service}_response.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Save parsed response
                parsed_output = output_dir / f"parsed_{input_file.stem}_{service}_response.jsonl"
                with open(parsed_output, 'w') as f:
                    if service == "openai":
                        content = result["response"].get("choices", [{}])[0].get("message", {}).get("content", "")
                        f.write(json.dumps({"content": content}) + '\n')
                    elif service == "deepl":
                        content = result["response"].get("translated_text", "")
                        f.write(json.dumps({"content": content}) + '\n')
                
                return result
            except Exception as e:
                logger.error(f"Error processing file {input_file} with {service}: {str(e)}")
                raise
            finally:
                # Clean up
                if similarity_checker:
                    similarity_checker.close()
                cache_manager.close()
        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {str(e)}")
            return {"error": str(e)}

    async def _process_batch_file(self, input_file: Union[str, Path], service: str, base_path: Union[str, Path], config: Dict) -> List[Dict]:
        """Process a batch file and return the results."""
        try:
            # Convert string paths back to Path objects
            input_file = Path(input_file)
            base_path = Path(base_path)
            
            # Initialize components for this process
            cache_manager = CacheManager(base_path, config)
            
            # Backup original file
            original_dir = base_path / "inputs" / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_file, original_dir / input_file.name)
            
            # Convert to JSONL and process
            jsonl_file = FileConverter.convert_to_jsonl(input_file, base_path)
            batch_size = config["openai"]["batch"]["batch_size"] if service == "openai" else config["deepl"].get("batch_size", 50)
            batch_files = self._split_into_batches(jsonl_file, batch_size)
            
            all_results = []
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = [json.loads(line) for line in f if line.strip()]
                    
                    # Send batch request
                    if service == "openai":
                        if self._should_use_file_batch(config):
                            # Use file batch API
                            # Prepare batch request
                            batch_request = {
                                "input_file_id": None,  # Will be set after file upload
                                "endpoint": "/v1/chat/completions",
                                "completion_window": "24h",
                                "metadata": {
                                    "description": f"Batch processing for {batch_file.name}"
                                }
                            }

                            # Prepare input file with custom_ids and method
                            with open(batch_file, 'r') as f:
                                input_data = [json.loads(line) for line in f if line.strip()]
                            
                            # Create a new JSONL file with custom_ids and method
                            temp_batch_file = batch_file.parent / f"temp_{batch_file.name}"
                            with open(temp_batch_file, 'w') as f:
                                for i, item in enumerate(input_data):
                                    # Create a new request object
                                    request = {
                                        'custom_id': f"request_{i}",
                                        'method': "POST",
                                        'url': "/v1/chat/completions"
                                    }
                                    
                                    # Get prompt content using the configured prompt field
                                    prompt_key = config["openai"]["prompt"]
                                    
                                    # Handle different input formats
                                    if isinstance(item, str):
                                        user_content = item
                                    elif isinstance(item, dict):
                                        if prompt_key in item:
                                            user_content = item[prompt_key]
                                        elif len(item) == 1:
                                            user_content = next(iter(item.values()))
                                        else:
                                            user_content = str(item)
                                    else:
                                        user_content = str(item)
                                    
                                    # Get system content if dynamic
                                    is_dynamic = config["openai"]["system_instruction"]["is_dynamic"]
                                    sys_prompt_key = config["openai"]["system_instruction"]["system_prompt"]
                                    system_content = item.get(sys_prompt_key, "You are a helpful assistant") if is_dynamic else "You are a helpful assistant"
                                    
                                    # Add the body
                                    request['body'] = {
                                        "model": config["openai"]["api_information"]["model"],
                                        "messages": [
                                            {"role": "system", "content": system_content},
                                            {"role": "user", "content": user_content}
                                        ],
                                        "temperature": config["openai"]["api_information"]["setting"]["temperature"],
                                        "max_tokens": config["openai"]["api_information"]["setting"]["max_tokens"]
                                    }
                                    
                                    f.write(json.dumps(request) + '\n')

                            # Upload input file
                            with open(temp_batch_file, 'rb') as f:
                                input_file = await self.openai_client.files.create(
                                    file=f,
                                    purpose='batch'
                                )
                            batch_request["input_file_id"] = input_file.id

                            # Create batch job
                            batch_job = await self.openai_client.batches.create(
                                **batch_request
                            )

                            # Clean up temporary file
                            temp_batch_file.unlink()

                            # Wait for processing
                            while True:
                                status = await self.openai_client.batches.retrieve(batch_job.id)
                                if status.status in ['completed', 'failed', 'expired', 'cancelled']:
                                    break
                                await asyncio.sleep(config["openai"]["batch"]["batch_check_time"])
                            
                            # Get batch results
                            if status.status == 'completed':
                                results = await self.openai_client.batches.retrieve(batch_job.id)
                                output_file_id = results.output_file_id
                                
                                # Download results
                                output_file = await self.openai_client.files.content(output_file_id)
                                
                                # Handle binary response
                                try:
                                    # First try to decode as text
                                    content = output_file.text
                                    try:
                                        # Try to parse as JSON
                                        batch_results = json.loads(content)
                                    except json.JSONDecodeError:
                                        # If not JSON, wrap in expected format
                                        batch_results = {
                                            "choices": [
                                                {
                                                    "message": {
                                                        "content": content
                                                    }
                                                }
                                            ]
                                        }
                                except Exception as e:
                                    # If text decoding fails, try to handle as binary
                                    try:
                                        content = output_file.content.decode('utf-8')
                                        try:
                                            batch_results = json.loads(content)
                                        except json.JSONDecodeError:
                                            batch_results = {
                                                "choices": [
                                                    {
                                                        "message": {
                                                            "content": content
                                                        }
                                                    }
                                                ]
                                            }
                                    except Exception as decode_error:
                                        logger.error(f"Error decoding binary response: {str(decode_error)}")
                                        batch_results = {
                                            "choices": [
                                                {
                                                    "message": {
                                                        "content": f"Error decoding response: {str(decode_error)}"
                                                    }
                                                }
                                            ]
                                        }
                            else:
                                batch_results = {
                                    "choices": [
                                        {
                                            "message": {
                                                "content": f"Batch job failed with status: {status.status}"
                                            }
                                        }
                                    ]
                                }
                        else:
                            # Use normal API with batching
                            # Prepare messages for batch
                            messages = []
                            for item in batch_data:
                                prompt_key = config["openai"]["prompt"]
                                is_dynamic = config["openai"]["system_instruction"]["is_dynamic"]
                                sys_prompt_key = config["openai"]["system_instruction"]["system_prompt"]
                                
                                user_content, system_content = find_prompt_content(
                                    item, prompt_key, sys_prompt_key, is_dynamic
                                )
                                messages.append({
                                    "role": "user",
                                    "content": user_content
                                })
                            
                            # Send batch request
                            response = await self.openai_client.chat.completions.create(
                                model=config["openai"]["api_information"]["model"],
                                messages=messages,
                                temperature=config["openai"]["api_information"]["setting"]["temperature"],
                                max_tokens=config["openai"]["api_information"]["setting"]["max_tokens"]
                            )
                            
                            # Handle different response types
                            if hasattr(response, 'model_dump'):
                                batch_results = response.model_dump()
                            elif hasattr(response, 'dict'):
                                batch_results = response.dict()
                            elif hasattr(response, 'json'):
                                batch_results = response.json()
                            else:
                                # Try to convert to dict if possible
                                try:
                                    batch_results = json.loads(str(response))
                                except (json.JSONDecodeError, TypeError):
                                    batch_results = {
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": str(response)
                                                }
                                            }
                                        ]
                                    }
                    elif service == "deepl":
                        batch_results = await self.deepl_client.translate_text(
                            [item.get("text", "") for item in batch_data],
                            target_lang=config["deepl"]["target_lang"]
                        )
                        batch_results = {
                            "responses": [
                                {
                                    "translated_text": t.text,
                                    "detected_source_lang": t.detected_source_lang
                                }
                                for t in batch_results
                            ]
                        }
                    
                    # Save batch results
                    output_dir = base_path / "outputs"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save raw response
                    raw_output = output_dir / f"{batch_file.stem}_batch_response.json"
                    with open(raw_output, 'w') as f:
                        # Save the complete raw response
                        if service == "openai":
                            if isinstance(output_file, (str, bytes)):
                                # For binary/text responses
                                f.write(str(output_file))
                            else:
                                # For object responses
                                try:
                                    if hasattr(output_file, 'model_dump'):
                                        json.dump(output_file.model_dump(), f, indent=2)
                                    elif hasattr(output_file, 'dict'):
                                        json.dump(output_file.dict(), f, indent=2)
                                    elif hasattr(output_file, 'json'):
                                        f.write(output_file.json())
                                    else:
                                        json.dump(str(output_file), f, indent=2)
                                except Exception as e:
                                    logger.error(f"Error saving raw response: {str(e)}")
                                    json.dump({"error": str(e), "raw_response": str(output_file)}, f, indent=2)
                        else:
                            json.dump(batch_results, f, indent=2)
                    
                    # Save formatted response
                    formatted_output = output_dir / f"{batch_file.stem}_batch_formatted.json"
                    with open(formatted_output, 'w') as f:
                        json.dump(batch_results, f, indent=2)
                    
                    # Save parsed response (content only)
                    parsed_output = output_dir / f"parsed_{batch_file.stem}_batch_response.jsonl"
                    with open(parsed_output, 'w') as f:
                        if service == "openai":
                            if isinstance(batch_results, dict):
                                choices = batch_results.get("choices", [])
                                for choice in choices:
                                    content = choice.get("message", {}).get("content", "")
                                    f.write(json.dumps({"content": content}) + '\n')
                            else:
                                # Handle case where results are not in expected format
                                f.write(json.dumps({"content": str(batch_results)}) + '\n')
                        elif service == "deepl":
                            for response in batch_results.get("responses", []):
                                content = response.get("translated_text", "")
                                f.write(json.dumps({"content": content}) + '\n')
                    
                    # Add batch results to all results
                    if service == "openai":
                        if isinstance(batch_results, dict):
                            all_results.extend(batch_results.get("choices", []))
                        else:
                            all_results.append({"message": {"content": str(batch_results)}})
                    else:
                        all_results.extend(batch_results.get("responses", []))
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_file} with {service}: {str(e)}")
                    raise
            
            # Combine results if configured
            if self._should_combine_batches(config):
                combined_output = base_path / "outputs" / f"{input_file.stem}_combined_response.jsonl"
                with open(combined_output, 'w') as f:
                    for response in all_results:
                        if isinstance(response, dict):
                            if service == "openai":
                                content = response.get("message", {}).get("content", "")
                            else:
                                content = response.get("translated_text", "")
                            f.write(json.dumps({"content": content}) + '\n')
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error processing batch {input_file.name}: {str(e)}")
            return [{"error": str(e)}]
        finally:
            # Clean up
            cache_manager.close()

    async def process_all_files(self, service: str = "openai") -> Dict[str, Union[Dict, List[Dict]]]:
        """Process all JSON and JSONL files in the input directory using async/await."""
        input_dir = self.base_path / "inputs"
        results = {}
        
        # Get all JSON and JSONL files
        input_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.jsonl"))
        
        if not input_files:
            logger.warning("No JSON or JSONL files found in input directory")
            return results
        
        logger.info(f"Found {len(input_files)} files to process with {service}")
        
        # Calculate number of concurrent tasks
        num_tasks = min(len(input_files), self.max_threads)
        logger.info(f"Using {num_tasks} concurrent tasks for processing")
        
        is_batch_active = self.openai_is_batch_active if service == "openai" else self.deepl_is_batch_active
        combine_batches = self.openai_combine_batches if service == "openai" else self.deepl_combine_batches
        
        # Process files concurrently
        tasks = []
        for input_file in input_files:
            if is_batch_active:
                task = asyncio.create_task(
                    self._process_batch_file(
                        str(input_file),
                        service,
                        str(self.base_path),
                        self.config
                    )
                )
            else:
                task = asyncio.create_task(
                    self._process_single_file(
                        str(input_file),
                        service,
                        str(self.base_path),
                        self.config
                    )
                )
            tasks.append((task, input_file))
        
        # Wait for all tasks to complete
        for task, input_file in tasks:
            try:
                result = await task
                if not combine_batches and is_batch_active:
                    # Save each batch result separately
                    for i, batch_result in enumerate(result):
                        output_file = self.base_path / "outputs" / f"{input_file.stem}_{service}_batch_{i}_response.json"
                        with open(output_file, 'w') as f:
                            json.dump(batch_result, f, indent=2)
                else:
                    # Save combined results
                    output_file = self.base_path / "outputs" / f"{input_file.stem}_{service}_responses.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
                results[input_file.name] = result
            except Exception as e:
                logger.error(f"Error processing {input_file.name} with {service}: {str(e)}")
                results[input_file.name] = {"error": str(e)}
        
        return results

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.cache_manager.get_stats()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache_manager.clear_all()
        self.logger.info("All caches cleared")

    def close(self) -> None:
        """Close the client and its resources."""
        self.cache_manager.close()
        self.logger.close()
        if self.similarity_checker:
            self.similarity_checker.close()

async def main():
    # Get the script's directory
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    # Prompt user to select existing task or create new one
    selected_task = SimpleClient.select_task()
    
    try:
        client = SimpleClient(config_path, selected_task)
        
        # Process with OpenAI if configured
        if client.openai_client:
            print("\nProcessing with OpenAI...")
            openai_results = await client.process_all_files(service="openai")
            print(f"OpenAI processing completed for {len(openai_results)} files.")
        
        # Process with DeepL if configured
        if client.deepl_client:
            print("\nProcessing with DeepL...")
            deepl_results = await client.process_all_files(service="deepl")
            print(f"DeepL processing completed for {len(deepl_results)} files.")
        
        print(f"\nCache stats: {client.get_cache_stats()}")
        
        # Print summary of results
        print("\nResults summary:")
        if client.openai_client:
            print("\nOpenAI Results:")
            for filename, result in openai_results.items():
                if "error" in result:
                    print(f"❌ {filename}: Failed - {result['error']}")
                else:
                    print(f"✅ {filename}: Success")
        
        if client.deepl_client:
            print("\nDeepL Results:")
            for filename, result in deepl_results.items():
                if "error" in result:
                    print(f"❌ {filename}: Failed - {result['error']}")
                else:
                    print(f"✅ {filename}: Success")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 