from pydantic import BaseModel
from typing import Dict, Any


class OpenAIBatch(BaseModel):
    is_batch_active: bool
    batch_check_time: int
    batch_size: int


class OpenAISetting(BaseModel):
    temperature: float
    max_tokens: int


class OpenAIAPIInfo(BaseModel):
    model: str
    setting: OpenAISetting


class OpenAIConfig(BaseModel):
    batch: OpenAIBatch
    prompt: str
    system_instruction: Dict[str, Any]
    api_key: str
    api_information: OpenAIAPIInfo
    bulk_save: int
    sleep_time: int
