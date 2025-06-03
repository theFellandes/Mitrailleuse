import pytest
import asyncio
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter
from mitrailleuse.infrastructure.adapters.deepseek_adapter import DeepSeekAdapter
from mitrailleuse.infrastructure.adapters.deepl_adapter import DeepLAdapter

import types

@pytest.mark.asyncio
async def test_openai_adapter_ping(monkeypatch):
    config = {"openai": {"api_key": "sk-xxx", "api_information": {"model": "gpt-4", "setting": {"temperature": 0.7, "max_tokens": 10}}}}
    adapter = OpenAIAdapter(config)
    async def dummy_create(*args, **kwargs):
        return {'choices':[{'message':{'content':'ok'}}]}
    dummy_completions = type("Completions", (), {"create": dummy_create})()
    dummy_chat = type("Chat", (), {"completions": dummy_completions})()
    dummy_client = type("Dummy", (), {"chat": dummy_chat})()
    monkeypatch.setattr(adapter, "client", dummy_client)
    result = await adapter.send_single({"messages": [{"role": "user", "content": "hi"}]})
    assert "choices" in result or "content" in result

# Add similar tests for DeepSeekAdapter and DeepLAdapter with dummy configs/mocks