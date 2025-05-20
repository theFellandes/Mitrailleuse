import json
from typing import Dict, Optional, Any

from pathlib import Path
from pydantic import BaseModel, Field

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


class SimilaritySettings(BaseModel):
    close_after_use: bool = True
    similarity_threshold: float = 0.8
    cooldown_period: int = 300
    max_recent_responses: int = 100


class SimilarityCheck(BaseModel):
    enabled: bool = False
    settings: SimilaritySettings = Field(default_factory=SimilaritySettings)


class SystemInstruction(BaseModel):
    is_dynamic: bool = False
    system_prompt: str = ""


class BatchConfig(BaseModel):
    is_batch_active: bool = False
    batch_size: int = 5
    batch_check_time: int = 120
    combine_batches: bool = False
    file_batch: bool = False


class APISettings(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 2000


class APIInformation(BaseModel):
    model: str
    setting: APISettings


class ProxyConfig(BaseModel):
    proxies_enabled: bool = False
    http: Optional[str] = None
    https: Optional[str] = None


class BaseAPIConfig(BaseModel):
    api_key: str
    prompt: str = "input_text"
    system_instruction: SystemInstruction = Field(default_factory=SystemInstruction)
    api_information: APIInformation
    batch: BatchConfig = Field(default_factory=BatchConfig)
    bulk_save: int = 10
    sleep_time: int = 0
    proxies: ProxyConfig = Field(default_factory=ProxyConfig)


class OpenAIConfig(BaseAPIConfig):
    pass


class DeepseekConfig(BaseAPIConfig):
    pass


class DeeplConfig(BaseModel):
    api_key: str
    target_lang: str = "EN-US"
    text: str = "text"


class GeneralConfig(BaseModel):
    verbose: bool
    sampling: Sampling
    logs: GeneralLoggingConfig
    multiprocessing_enabled: bool
    num_processes: int
    process_cap_percentage: int = 75
    db: DBConfig
    similarity_check: SimilarityCheck = Field(default_factory=SimilarityCheck)


class Config(BaseModel):
    task_name: str
    user_id: str
    openai: OpenAIConfig
    deepseek: DeepseekConfig
    deepl: DeeplConfig
    general: GeneralConfig

    def deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
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
            json.dump(config.model_dump(), file, indent=4, ensure_ascii=False)

    def find_task_names(self, data: Any, parent_key: str = '', result: list = None) -> list:
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
