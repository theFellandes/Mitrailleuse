import pytest
from mitrailleuse.infrastructure.adapters.deepl_adapter import DeepLAdapter

class DummyDeeplConfig:
    api_key = "dummy"
    text = "text"
    target_lang = "EN"

@pytest.mark.asyncio
async def test_deepl_adapter_send_single(monkeypatch):
    adapter = DeepLAdapter(DummyDeeplConfig())
    # Patch the _client.post to return a dummy response
    class DummyResponse:
        def raise_for_status(self): pass
        def json(self): return {"translations": [{"text": "hello"}]}
    async def dummy_post(*args, **kwargs): return DummyResponse()
    monkeypatch.setattr(adapter._client, "post", dummy_post)
    payload = {"text": "Hello", "target_lang": "EN"}
    result = await adapter.send_single(payload)
    assert "translations" in result
