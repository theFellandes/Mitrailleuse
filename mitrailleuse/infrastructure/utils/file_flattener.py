import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Any
from datetime import datetime
import shutil
from .logger import get_logger

logger = logging.getLogger(__name__)

class FileFlattener:
    """Utility class for flattening nested JSON structures."""

    def __init__(self, base_path: Path, config: Dict):
        self.base_path = base_path
        self.config = config
        self.prompt_key = self._get_prompt_key()
        self.instruction_key = self._get_instruction_key()
        self.logger = get_logger("file_flattener", config)

    def _get_prompt_key(self) -> str:
        """Get the prompt key from config."""
        try:
            return self.config["openai"]["prompt"]
        except (KeyError, TypeError):
            return "input_text"  # Default fallback

    def _get_instruction_key(self) -> str:
        """Get the instruction key from config."""
        try:
            return self.config["openai"]["system_instruction"]["system_prompt"]
        except (KeyError, TypeError):
            return "instructions"  # Default fallback

    def _backup_input_file(self, input_file: Path) -> None:
        """Backup the input file to the backup directory."""
        backup_dir = self.base_path / "inputs" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{input_file.stem}_{timestamp}{input_file.suffix}"
        shutil.copy2(input_file, backup_file)
        self.logger.info(f"Backed up input file to: {backup_file}")

    def _flatten_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a single item while preserving all fields."""
        result = {}
        
        # Always preserve the original item
        result.update(item)
        
        # If the item has nested prompts, flatten them
        if "prompts" in item and isinstance(item["prompts"], list):
            flattened_prompts = []
            for prompt in item["prompts"]:
                if isinstance(prompt, dict):
                    # Preserve all fields in the prompt
                    flattened_prompts.append(prompt)
            result["prompts"] = flattened_prompts
        
        return result

    def flatten_file(self, input_file: Path) -> Path:
        """Flatten a JSON file while preserving all fields."""
        try:
            # Backup original file
            self._backup_input_file(input_file)

            # Read the input file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both array and object formats
            if isinstance(data, list):
                # If it's already a list of items, flatten each item
                flattened_data = [self._flatten_item(item) for item in data]
            elif isinstance(data, dict):
                # If it's an object with prompts, flatten it
                flattened_data = [self._flatten_item(data)]
            else:
                raise ValueError(f"Unsupported data format in {input_file}")

            # Create output file path
            output_file = self.base_path / "inputs" / f"{input_file.stem}_flattened{input_file.suffix}"
            
            # Write flattened data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(flattened_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Flattened {input_file} to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error flattening file {input_file}: {str(e)}")
            raise

    def flatten_all_files(self) -> List[Path]:
        """
        Flatten all JSON and JSONL files in the input directory.
        
        Returns:
            List of paths to flattened files
        """
        input_dir = self.base_path / "inputs"
        flattened_files = []
        
        # Get all JSON and JSONL files
        input_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.jsonl"))
        
        for input_file in input_files:
            try:
                flattened_file = self.flatten_file(input_file)
                flattened_files.append(flattened_file)
            except Exception as e:
                self.logger.error(f"Error flattening file {input_file}: {str(e)}")
        
        return flattened_files 