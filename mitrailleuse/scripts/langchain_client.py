"""
Mitrailleuse LangChain Client - A lightweight client using LangChain for API interactions

Requirements:
pip install langchain langchain-openai langchain-community python-dotenv
"""
import json
import os
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks import FileCallbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LangChainClient:
    def __init__(self, config_path: Union[str, Path], task_path: Optional[Union[str, Path]] = None):
        """Initialize the client with configuration."""
        self.config_path = Path(config_path)
        self.task_path = Path(task_path) if task_path else None
        
        # Load environment variables
        load_dotenv()
        
        # Load initial config
        self.config = self._load_config(self.config_path)
        self.user_id = self.config.get("user_id", "default_user")
        self.task_name = self.config["task_name"]
        
        # Set up task directory
        if self.task_path:
            if not self.task_path.exists():
                raise ValueError(f"Specified task path does not exist: {task_path}")
            self.base_path = self.task_path
            self.config = self._load_config(self.base_path / "config" / "config.json")
        else:
            self.base_path = self._create_task_directory()
            self.config = self._load_config(self.base_path / "config" / "config.json")
        
        # Initialize LangChain components
        self._setup_langchain()
        
        # Create necessary directories
        self._create_task_directories()
        self._check_input_files()

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
        shutil.copy2(self.config_path, config_dir / "config.json")
        
        logger.info(f"Created new task directory: {task_dir}")
        return task_dir

    def _create_task_directories(self):
        """Create necessary directories for the task."""
        dirs = ["inputs", "outputs", "logs", "cache"]
        for dir_name in dirs:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)

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

    def _setup_langchain(self):
        """Set up LangChain components."""
        # Get model configuration
        model_name = self.config["openai"]["api_information"]["model"]
        temperature = self.config["openai"]["api_information"]["setting"]["temperature"]
        max_tokens = self.config["openai"]["api_information"]["setting"]["max_tokens"]

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Set up the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            ("user", "{user_message}")
        ])

        # Set up the chain
        self.chain = (
            {"system_message": RunnablePassthrough(), "user_message": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    async def _process_single_file(self, input_file: Path) -> Dict:
        """Process a single file using LangChain."""
        try:
            # Read input file
            with open(input_file, 'r') as f:
                input_data = json.load(f)
            
            if not isinstance(input_data, list):
                input_data = [input_data]

            results = []
            for item in input_data:
                # Get prompt content
                prompt_key = self.config["openai"]["prompt"]
                if isinstance(item, str):
                    user_content = item
                elif isinstance(item, dict):
                    user_content = item.get(prompt_key, str(item))
                else:
                    user_content = str(item)

                # Get system message
                is_dynamic = self.config["openai"]["system_instruction"]["is_dynamic"]
                sys_prompt_key = self.config["openai"]["system_instruction"]["system_prompt"]
                system_content = item.get(sys_prompt_key, "You are a helpful assistant") if is_dynamic else "You are a helpful assistant"

                # Process with LangChain
                start_time = time.time()
                response = await self.chain.ainvoke({
                    "system_message": system_content,
                    "user_message": user_content
                })
                duration = time.time() - start_time

                # Format result
                result = {
                    "input": item,
                    "response": {
                        "choices": [{
                            "message": {
                                "content": response
                            }
                        }]
                    },
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

            # Save results
            output_dir = self.base_path / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save raw response
            raw_output = output_dir / f"{input_file.stem}_raw_response.json"
            with open(raw_output, 'w') as f:
                json.dump(results, f, indent=2)

            # Save formatted response
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "system": result["input"].get("system", "You are a helpful assistant"),
                    "user": result["input"].get(prompt_key, str(result["input"])),
                    "assistant": result["response"]["choices"][0]["message"]["content"]
                })

            formatted_output = output_dir / f"{input_file.stem}_formatted_response.json"
            with open(formatted_output, 'w') as f:
                json.dump(formatted_results, f, indent=2)

            # Save parsed response
            parsed_output = output_dir / f"parsed_{input_file.stem}_response.jsonl"
            with open(parsed_output, 'w') as f:
                for result in formatted_results:
                    f.write(json.dumps({"content": result["assistant"]}) + '\n')

            return results

        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {str(e)}")
            return {"error": str(e)}

    async def process_all_files(self) -> Dict[str, Union[Dict, List[Dict]]]:
        """Process all JSON and JSONL files in the input directory."""
        input_dir = self.base_path / "inputs"
        results = {}
        
        # Get all JSON and JSONL files
        input_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.jsonl"))
        
        if not input_files:
            logger.warning("No JSON or JSONL files found in input directory")
            return results
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process files concurrently
        tasks = []
        for input_file in input_files:
            task = asyncio.create_task(self._process_single_file(input_file))
            tasks.append((task, input_file))
        
        # Wait for all tasks to complete
        for task, input_file in tasks:
            try:
                result = await task
                results[input_file.name] = result
            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {str(e)}")
                results[input_file.name] = {"error": str(e)}
        
        return results

async def main():
    # Get the script's directory
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    try:
        client = LangChainClient(config_path)
        
        # Process all files
        print("\nProcessing files...")
        results = await client.process_all_files()
        
        # Print summary of results
        print("\nResults summary:")
        for filename, result in results.items():
            if "error" in result:
                print(f"❌ {filename}: Failed - {result['error']}")
            else:
                print(f"✅ {filename}: Success")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 