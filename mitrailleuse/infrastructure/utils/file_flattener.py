import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil
from .logger import get_logger

class FileFlattener:
    """Utility class for flattening input files based on configuration."""

    def __init__(self, base_path: Path, config: Dict[str, Any]):
        """
        Initialize the file flattener.
        
        Args:
            base_path: Base path where files will be stored
            config: Configuration dictionary
        """
        self.base_path = base_path
        self.config = config
        self.logger = get_logger("file_flattener", config)

    def _backup_input_file(self, input_file: Path) -> None:
        """Backup the input file to the backup directory."""
        backup_dir = self.base_path / "inputs" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{input_file.stem}_{timestamp}{input_file.suffix}"
        shutil.copy2(input_file, backup_file)
        self.logger.info(f"Backed up input file to: {backup_file}")

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[Any]:
        """
        Get a nested value from a dictionary using dot notation.
        
        Args:
            data: Input dictionary
            path: Dot-notation path to the value
            
        Returns:
            Value if found, None otherwise
        """
        try:
            for key in path.split('.'):
                data = data[key]
            return data
        except (KeyError, TypeError):
            return None

    def _flatten_item(self, item: Dict[str, Any], prompt_key: str, system_prompt_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Flatten a single item based on configuration.
        
        Args:
            item: Input item to flatten
            prompt_key: Key for the prompt field
            system_prompt_key: Optional key for the system prompt field
            
        Returns:
            Flattened item
        """
        flattened = {}
        
        # Get prompt value
        prompt_value = self._get_nested_value(item, prompt_key)
        if prompt_value is not None:
            flattened[prompt_key] = prompt_value
        
        # Get system prompt value if configured
        if system_prompt_key:
            system_prompt_value = self._get_nested_value(item, system_prompt_key)
            if system_prompt_value is not None:
                flattened[system_prompt_key] = system_prompt_value
        
        return flattened

    def flatten_file(self, input_file: Path) -> Path:
        """
        Flatten an input file based on configuration.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            Path to the flattened file
        """
        # Backup original file
        self._backup_input_file(input_file)
        
        # Get configuration
        prompt_key = self.config.get("openai", {}).get("prompt", "input_text")
        system_prompt_key = None
        if self.config.get("openai", {}).get("system_instruction", {}).get("is_dynamic", False):
            system_prompt_key = self.config["openai"]["system_instruction"].get("system_prompt", "instructions")
        
        # Read input file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Handle both single items and lists
        if not isinstance(data, list):
            data = [data]
        
        # Flatten each item
        flattened_data = [
            self._flatten_item(item, prompt_key, system_prompt_key)
            for item in data
        ]
        
        # Create flattened file
        flattened_dir = self.base_path / "inputs" / "flattened"
        flattened_dir.mkdir(parents=True, exist_ok=True)
        
        flattened_file = flattened_dir / f"{input_file.stem}_flattened{input_file.suffix}"
        with open(flattened_file, 'w') as f:
            if len(flattened_data) == 1:
                json.dump(flattened_data[0], f, indent=2)
            else:
                json.dump(flattened_data, f, indent=2)
        
        self.logger.info(f"Created flattened file: {flattened_file}")
        return flattened_file

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