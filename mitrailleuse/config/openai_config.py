from pydantic import BaseModel
from typing import Dict, Any

class OpenAISetting(BaseModel):
    temperature: float
    max_tokens: int

class OpenAIAPIInfo(BaseModel):
    model: str
    setting: OpenAISetting

class OpenAIProxies(BaseModel):
    proxies_enabled: bool
    http: str
    https: str

class OpenAIConfig(BaseModel):
    batch: Dict[str, Any]
    prompt: str
    system_instruction: Dict[str, Any]
    api_key: str
    api_information: OpenAIAPIInfo
    bulk_save: int
    sleep_time: int
    proxies: OpenAIProxies