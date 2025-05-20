import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponseFormatter:
    def __init__(self, user_id: str, task_name: str, task_path: Optional[Union[str, Path]] = None):
        """Initialize the formatter with user and task information."""
        self.user_id = user_id
        self.task_name = task_name
        self.script_dir = Path(__file__).parent.absolute()
        
        if task_path:
            self.task_path = Path(task_path)
        else:
            # Find the task directory in the scripts/tasks directory
            tasks_dir = self.script_dir / "tasks" / self.user_id
            
            if not tasks_dir.exists():
                raise ValueError(f"No task directory found for user {self.user_id}")
            
            # Get all task directories for this user and task name
            task_dirs = sorted(
                [d for d in tasks_dir.glob(f"{self.task_name}*") if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )
            
            if not task_dirs:
                raise ValueError(f"No task directory found for {self.task_name}")
            
            self.task_path = task_dirs[0]
            logger.info(f"Using existing task directory: {self.task_path}")
        
        self.config_path = self.task_path / "config" / "config.json"
        self.inputs_dir = self.task_path / "inputs"
        self.outputs_dir = self.task_path / "outputs"
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def format_all_responses(self) -> None:
        """Format all raw responses in the outputs directory."""
        # Get all raw response files
        raw_responses = list(self.outputs_dir.glob("*_raw_response.json"))
        
        if not raw_responses:
            logger.warning(f"No raw response files found in {self.outputs_dir}")
            return
        
        logger.info(f"Found {len(raw_responses)} raw response files to format")
        
        for raw_response in raw_responses:
            try:
                # Create formatted output path
                formatted_output = self.outputs_dir / f"{raw_response.stem.replace('_raw_response', '_formatted_response')}.json"
                
                # Format the response
                formatted_results = self.format_batch_response(raw_response, formatted_output)
                
                # Create parsed output
                parsed_output = self.outputs_dir / f"parsed_{raw_response.stem.replace('_raw_response', '')}.jsonl"
                with open(parsed_output, 'w') as f:
                    for result in formatted_results:
                        f.write(json.dumps({"content": result["assistant"]}) + '\n')
                
                logger.info(f"Formatted {raw_response.name} → {formatted_output.name}")
                
            except Exception as e:
                logger.error(f"Error formatting {raw_response.name}: {str(e)}")

    def format_batch_response(self, raw_response_path: Union[str, Path], output_path: Union[str, Path] = None) -> List[Dict]:
        """
        Format the raw batch response into the desired format with system, user, and assistant messages.
        
        Args:
            raw_response_path: Path to the raw response JSON file
            output_path: Optional path to save the formatted response
        
        Returns:
            List of formatted responses
        """
        try:
            # Convert paths to Path objects
            raw_response_path = Path(raw_response_path)
            
            # Load raw response
            with open(raw_response_path, 'r') as f:
                raw_results = json.load(f)
            
            # Get config values
            prompt_key = self.config["openai"]["prompt"]
            is_dynamic = self.config["openai"]["system_instruction"]["is_dynamic"]
            sys_prompt_key = self.config["openai"]["system_instruction"]["system_prompt"]
            
            # Get the corresponding input file
            input_file_name = raw_response_path.stem.replace('_raw_response', '')
            if '_batch_' in input_file_name:
                # For batch files, get the original batch file
                batch_num = input_file_name.split('_batch_')[1].split('_')[0]
                input_file = self.inputs_dir / f"batch_{batch_num}.jsonl"
            else:
                input_file = self.inputs_dir / f"{input_file_name}.jsonl"
            
            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                return []
            
            # Read input data
            with open(input_file, 'r') as f:
                input_data = [json.loads(line) for line in f if line.strip()]
            
            # Format results
            formatted_results = []
            for i, item in enumerate(raw_results):
                try:
                    # Get user content from input data
                    if i < len(input_data):
                        input_item = input_data[i]
                        if isinstance(input_item, str):
                            user_content = input_item
                        elif isinstance(input_item, dict):
                            if prompt_key in input_item:
                                user_content = input_item[prompt_key]
                            elif len(input_item) == 1:
                                user_content = next(iter(input_item.values()))
                            else:
                                user_content = str(input_item)
                        else:
                            user_content = str(input_item)
                    else:
                        user_content = "Error: Input data not found"
                    
                    # Get system content
                    system_content = input_data[i].get(sys_prompt_key, "You are a helpful assistant") if is_dynamic else "You are a helpful assistant"
                    
                    # Get assistant content
                    message = item["response"]["body"]["choices"][0]["message"]
                    assistant_content = message["content"]
                    
                    formatted_results.append({
                        "system": system_content,
                        "user": user_content,
                        "assistant": assistant_content
                    })
                except (KeyError, IndexError) as e:
                    logger.error(f"Error formatting response: {str(e)}")
                    formatted_results.append({
                        "system": "error",
                        "user": "error",
                        "assistant": f"Error formatting response: {str(e)}"
                    })
            
            # Save formatted results if output path is provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(formatted_results, f, indent=2)
                logger.info(f"Formatted response saved to: {output_path}")
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error in format_batch_response: {str(e)}")
            raise

def main():
    """Standalone script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Format raw batch responses in a task directory')
    parser.add_argument('user_id', help='User ID')
    parser.add_argument('task_name', help='Task name (e.g., openai_test_20250520_141029)')
    parser.add_argument('--task_path', help='Path to existing task directory (optional)')
    
    args = parser.parse_args()
    
    try:
        formatter = ResponseFormatter(args.user_id, args.task_name, args.task_path)
        formatter.format_all_responses()
        print(f"Successfully formatted responses in {formatter.task_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 