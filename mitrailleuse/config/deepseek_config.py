from typing import Dict, Any

from pydantic import BaseModel


class DeepseekBatch(BaseModel):
    is_batch_active: bool
    batch_check_time: int
    batch_size: int


class DeepseekSetting(BaseModel):
    temperature: float
    max_tokens: int


class DeepseekAPIInfo(BaseModel):
    model: str
    setting: DeepseekSetting


class DeepseekProxies(BaseModel):
    proxies_enabled: bool
    http: str
    https: str


class DeepseekConfig(BaseModel):
    batch: DeepseekBatch
    prompt: str
    system_instruction: Dict[str, Any]
    api_key: str
    api_information: DeepseekAPIInfo
    bulk_save: int
    sleep_time: int
    proxies: DeepseekProxies
