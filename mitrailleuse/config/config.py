import json
from typing import Dict

from pathlib import Path
from pydantic import BaseModel

from mitrailleuse.config.deepl_config import DeeplConfig
from mitrailleuse.config.deepseek_config import DeepseekConfig
from mitrailleuse.config.openai_config import OpenAIConfig


class GeneralLoggingConfig(BaseModel):
    log_file: str
    log_to_file: bool
    log_to_db: bool
    log_buffer_size: int


class DBConfig(BaseModel):
    postgres: Dict[str, str]


class Sampling(BaseModel):
    enable_sampling: bool
    sample_size: int


class GeneralConfig(BaseModel):
    verbose: bool
    sampling: Sampling
    logs: GeneralLoggingConfig
    multiprocessing_enabled: bool
    num_processes: int
    db: DBConfig
    check_similarity: bool
    similarity_stop_action: bool


class Config(BaseModel):
    task_name: str
    openai: OpenAIConfig
    deepseek: DeepseekConfig
    deepl: DeeplConfig
    general: GeneralConfig

    def deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update a dictionary with another dictionary."""
        result = base_dict.copy()

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_update(result[key], value)
            else:
                result[key] = value

        return result

    def update(self, update_data: dict):
        """
        Updates the model with the provided data, preserving existing values for unset fields.
        Handles nested updates correctly.
        """
        # Convert the current model to a dictionary with all fields included
        current_data = self.model_dump()

        # Deep update the current data with the update data
        updated_data = self.deep_update(current_data, update_data)

        # Make sure the task_name is not None
        if "task_name" not in updated_data or updated_data["task_name"] is None:
            updated_data["task_name"] = self.task_name

        # Create a new instance with the updated data
        return self.__class__(**updated_data)

    @classmethod
    def read(cls, path: Path) -> "Config":
        return cls.model_validate_json(path.read_text())

    @staticmethod
    def write(cfg: "Config", path: Path) -> None:
        """
        Persist *exactly* what Pydantic would emit as canonical JSON.
        Pydantic v2 no longer supports the ``ensure_ascii`` kwarg.
        """
        path.write_text(cfg.model_dump_json(indent=2))

    @staticmethod
    def write_old_version(config, config_file: str, encoding: str = 'utf-8') -> None:
        """
        Writes a Config instance to the specified config file in JSON format.
        """
        with open(config_file, 'w', encoding=encoding) as file:
            json.dump(config.dict(), file, indent=4, ensure_ascii=False)

    def find_task_names(self, data, parent_key='', result=None):
        if result is None:
            result = []

        if isinstance(data, dict):  # If the current element is a dictionary
            for key, value in data.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                self.find_task_names(value, new_key, result)
        elif isinstance(data, list):  # If the current element is a list
            for index, item in enumerate(data):
                new_key = f"{parent_key}_{index}"
                self.find_task_names(item, new_key, result)
        else:  # If it's a leaf element, check if it's 'task_name'
            if parent_key.endswith('task_name'):  # Check if the key ends with 'task_name'
                result.append((parent_key, data))

        return result
