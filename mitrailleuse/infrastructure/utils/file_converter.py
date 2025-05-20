import json
import shutil
import random
from datetime import datetime
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class FileConverter:
    """Utility class for handling file format conversion and backup operations."""

    @staticmethod
    def backup_file(input_file: Path, base_path: Path) -> None:
        """
        Backup the input file to the backup directory.
        
        Args:
            input_file: Path to the file to backup
            base_path: Base path where the backup directory will be created
        """
        backup_dir = base_path / "inputs" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{input_file.stem}_{timestamp}{input_file.suffix}"
        shutil.copy2(input_file, backup_file)
        log.info(f"Backed up input file to: {backup_file}")

    @staticmethod
    def _apply_sampling(data: list, sample_size: int) -> list:
        """
        Apply sampling to the data if it exceeds the sample size.
        
        Args:
            data: List of data items
            sample_size: Maximum number of items to keep
            
        Returns:
            Sampled list of data items
        """
        if len(data) > sample_size:
            log.info(f"Sampling enabled: reducing {len(data)} items to {sample_size} items")
            return random.sample(data, sample_size)
        return data

    @staticmethod
    def convert_to_jsonl(input_file: Path, base_path: Path, config: dict = None) -> Path:
        """
        Convert JSON file to JSONL format if needed and backup original.
        
        Args:
            input_file: Path to the input file
            base_path: Base path for backup directory
            config: Optional configuration dictionary containing sampling settings
            
        Returns:
            Path to the converted file (same as input if already JSONL)
        """
        # Backup the original file
        FileConverter.backup_file(input_file, base_path)
        
        if input_file.suffix == '.jsonl':
            # If it's already JSONL, check if we need to sample
            if config and config.get("general", {}).get("sampling", {}).get("enable_sampling", False):
                sample_size = config["general"]["sampling"]["sample_size"]
                with open(input_file, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                data = FileConverter._apply_sampling(data, sample_size)
                output_file = input_file.with_name(f"{input_file.stem}_sampled{input_file.suffix}")
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
                log.info(f"Applied sampling to {input_file}")
                return output_file
            return input_file

        output_file = input_file.with_suffix('.jsonl')
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]

        # Apply sampling if enabled in config
        if config and config.get("general", {}).get("sampling", {}).get("enable_sampling", False):
            sample_size = config["general"]["sampling"]["sample_size"]
            data = FileConverter._apply_sampling(data, sample_size)

        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        log.info(f"Converted {input_file} to JSONL format")
        return output_file

    @staticmethod
    def convert_to_json(input_file: Path, base_path: Path, config: dict = None) -> Path:
        """
        Convert JSONL file to JSON format if needed and backup original.
        
        Args:
            input_file: Path to the input file
            base_path: Base path for backup directory
            config: Optional configuration dictionary containing sampling settings
            
        Returns:
            Path to the converted file (same as input if already JSON)
        """
        # Backup the original file
        FileConverter.backup_file(input_file, base_path)
        
        if input_file.suffix == '.json':
            # If it's already JSON, check if we need to sample
            if config and config.get("general", {}).get("sampling", {}).get("enable_sampling", False):
                sample_size = config["general"]["sampling"]["sample_size"]
                with open(input_file, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                data = FileConverter._apply_sampling(data, sample_size)
                output_file = input_file.with_name(f"{input_file.stem}_sampled{input_file.suffix}")
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                log.info(f"Applied sampling to {input_file}")
                return output_file
            return input_file

        output_file = input_file.with_suffix('.json')
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]

        # Apply sampling if enabled in config
        if config and config.get("general", {}).get("sampling", {}).get("enable_sampling", False):
            sample_size = config["general"]["sampling"]["sample_size"]
            data = FileConverter._apply_sampling(data, sample_size)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        log.info(f"Converted {input_file} to JSON format")
        return output_file

    @staticmethod
    def ensure_directory_structure(base_path: Path) -> None:
        """
        Ensure all necessary directories exist.
        
        Args:
            base_path: Base path where directories will be created
        """
        for dir_name in ["inputs", "outputs", "logs", "inputs/backup"]:
            (base_path / dir_name).mkdir(parents=True, exist_ok=True) 