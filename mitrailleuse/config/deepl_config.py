from pydantic import BaseModel


class DeeplConfig(BaseModel):
    api_key: str
    target_lang: str
    text: str
