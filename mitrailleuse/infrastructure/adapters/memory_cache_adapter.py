import hashlib
import json
from typing import Any, Dict, Optional
from mitrailleuse.application.ports.cache_port import CachePort
from mitrailleuse.infrastructure.logging.logger import get_logger

log = get_logger(__name__)

class MemoryCache(CachePort):
    """In-memory cache implementation that hashes requests and caches responses."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute a deterministic hash of the request data."""
        # Sort keys to ensure consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if key in self._cache:
            self._hits += 1
            log.info(f"Cache hit for key: {key}")
            return self._cache[key]
        self._misses += 1
        log.info(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        self._cache[key] = value
        log.info(f"Cached value for key: {key}")
    
    def get_or_set(self, data: Dict[str, Any], compute_value: callable) -> Any:
        """Get from cache or compute and cache the value."""
        cache_key = self._compute_hash(data)
        cached_value = self.get(cache_key)
        
        if cached_value is not None:
            log.info(f"Using cached response for request: {cache_key}")
            return cached_value
        
        # Compute new value
        value = compute_value()
        self.set(cache_key, value)
        return value
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        log.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache)
        } 