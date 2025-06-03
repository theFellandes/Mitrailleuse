import pytest
from mitrailleuse.infrastructure.adapters.deepseek_adapter import DeepSeekAdapter

@pytest.mark.asyncio
async def test_deepseek_adapter_send_single(monkeypatch):
    config = {"deepseek": {"api_key": "dummy"}}
    adapter = DeepSeekAdapter(config)
    class DummyResponse:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "ok"}}]}
    async def dummy_post(*args, **kwargs): return DummyResponse()
    monkeypatch.setattr(adapter._client, "post", dummy_post)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    result = await adapter.send_single(payload)
    assert "choices" in result
