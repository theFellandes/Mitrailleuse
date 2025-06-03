import pytest
from mitrailleuse.application.ports.api_port import APIPort
from mitrailleuse.application.ports.cache_port import CachePort

def test_api_port_abstract_methods():
    class Dummy(APIPort):
        pass
    with pytest.raises(TypeError):
        Dummy()

def test_cache_port_abstract_methods():
    class Dummy(CachePort):
        pass
    with pytest.raises(TypeError):
        Dummy()
