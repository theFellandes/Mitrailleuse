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
            prompt_key = self.config["openai"]["prompt"]
            is_dynamic = self.config["openai"]["system_instruction"]["is_dynamic"]
            sys_prompt_key = self.config["openai"]["system_instruction"]["system_prompt"]

            user_content = input_data.get(prompt_key)
            if not user_content:
                raise ValueError(f"Required prompt field '{prompt_key}' not found in input")

            if is_dynamic:
                system_content = input_data.get(sys_prompt_key, "You are a helpful assistant")
            else:
                system_content = "You are a helpful assistant"

            return {
                "model": self.config["openai"]["api_information"]["model"],
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                "temperature": self.config["openai"]["api_information"]["setting"]["temperature"],
                "max_tokens": self.config["openai"]["api_information"]["setting"]["max_tokens"]
            }
        elif service == "deepl":
            text = input_data.get("text")
            if not text:
                raise ValueError("Required 'text' field not found in input")
            
            target_lang = input_data.get("target_lang", self.config["deepl"].get("default_target_lang", "EN-US"))
            source_lang = input_data.get("source_lang", self.config["deepl"].get("default_source_lang"))
            
            return {
                "text": text,
                "target_lang": target_lang,
                "source_lang": source_lang
            }
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

    async def _process_single_file(self, input_file: Union[str, Path], service: str, base_path: Union[str, Path], config: Dict) -> Dict:
        """Process a single file and return the result."""
        try:
            # Convert string paths back to Path objects
            input_file = Path(input_file)
            base_path = Path(base_path)
            
            # Initialize components for this process
            cache_manager = CacheManager(base_path, config)
            file_flattener = FileFlattener(base_path, config)
            
            # Initialize API client for this process
            if service == "openai":
                api_client = AsyncOpenAI(api_key=config["openai"]["api_key"])
            elif service == "deepl":
                api_client = deepl.Translator(config["deepl"]["api_key"])
            
            # Flatten the input file first
            flattened_file = file_flattener.flatten_file(input_file)
            
            # Convert to JSONL for processing
            jsonl_file = FileConverter.convert_to_jsonl(flattened_file, base_path)
            
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
                    response = await api_client.chat.completions.create(**request_body)
                    result = {
                        "input": input_data,
                        "response": response.model_dump(),
                        "timestamp": datetime.now().isoformat()
                    }
                elif service == "deepl":
                    translation = api_client.translate_text(
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
                
                # Save response
                output_file = base_path / "outputs" / f"{input_file.stem}_{service}_response.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
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
            file_flattener = FileFlattener(base_path, config)
            
            # Initialize API client for this process
            if service == "openai":
                api_client = AsyncOpenAI(api_key=config["openai"]["api_key"])
            elif service == "deepl":
                api_client = deepl.Translator(config["deepl"]["api_key"])
            
            # Flatten the input file first
            flattened_file = file_flattener.flatten_file(input_file)
            
            # Convert to JSONL and process
            jsonl_file = FileConverter.convert_to_jsonl(flattened_file, base_path)
            batch_size = config["openai"]["batch"]["batch_size"] if service == "openai" else config["deepl"].get("batch_size", 50)
            batch_files = self._split_into_batches(jsonl_file, batch_size)
            
            results = []
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = [json.loads(line) for line in f if line.strip()]
                    
                    batch_results = []
                    for item in batch_data:
                        # Check cache first
                        cached_response = cache_manager.get(item, service)
                        if cached_response:
                            batch_results.append(cached_response)
                            continue

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
                        request_body = self._build_request_body(item, service)
                        if service == "openai":
                            response = await api_client.chat.completions.create(**request_body)
                            result = {
                                "input": item,
                                "response": response.model_dump(),
                                "timestamp": datetime.now().isoformat()
                            }
                        elif service == "deepl":
                            translation = api_client.translate_text(
                                request_body["text"],
                                target_lang=request_body["target_lang"],
                                source_lang=request_body["source_lang"]
                            )
                            result = {
                                "input": item,
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
                        cache_manager.set(item, result, service)
                        batch_results.append(result)
                        
                        # Clean up similarity checker after each item
                        if similarity_checker:
                            similarity_checker.close()
                    
                    # Save batch results
                    output_file = base_path / "outputs" / f"{batch_file.stem}_{service}_responses.json"
                    with open(output_file, 'w') as f:
                        json.dump(batch_results, f, indent=2)
                    
                    results.extend(batch_results)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_file} with {service}: {str(e)}")
                    raise
            return results
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