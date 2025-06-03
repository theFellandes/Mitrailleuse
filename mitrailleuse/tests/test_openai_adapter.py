import pytest
from mitrailleuse.infrastructure.adapters.openai_adapter import OpenAIAdapter

@pytest.mark.asyncio
async def test_openai_adapter_send_single(monkeypatch):
    config = {"openai": {"api_key": "dummy", "api_information": {"model": "gpt-4", "setting": {"temperature": 0.7, "max_tokens": 10}}}}
    adapter = OpenAIAdapter(config)
    class DummyChat:
        class Completions:
            @staticmethod
            async def create(**kwargs):
                return {"choices": [{"message": {"content": "ok"}}]}
        completions = Completions()
    class DummyClient:
        chat = DummyChat()
    monkeypatch.setattr(adapter, "client", DummyClient())
    result = await adapter.send_single({"messages": [{"role": "user", "content": "hi"}]})
    assert "choices" in result or "content" in result
