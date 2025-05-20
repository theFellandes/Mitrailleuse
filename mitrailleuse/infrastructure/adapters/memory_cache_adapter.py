import hashlib
import json
from typing import Any, Dict, Optional, Callable
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
    
    def get(self, key: Any) -> Optional[Any]:
        """Get a value from the cache."""
        cache_key = self._generate_key(key)
        if cache_key in self._cache:
            self._hits += 1
            log.info(f"Cache hit for key: {cache_key}")
            return self._cache[cache_key]
        self._misses += 1
        log.info(f"Cache miss for key: {cache_key}")
        return None
    
    def set(self, key: Any, value: Any) -> None:
        """Set a value in the cache."""
        cache_key = self._generate_key(key)
        self._cache[cache_key] = value
        log.info(f"Cached value for key: {cache_key}")
    
    def has(self, key: Any) -> bool:
        """Check if a key exists in the cache."""
        cache_key = self._generate_key(key)
        return cache_key in self._cache
    
    def get_or_set(self, key: Any, getter: Callable[[], Any]) -> Any:
        """Get a value from cache or compute and set it."""
        if self.has(key):
            return self.get(key)
        value = getter()
        self.set(key, value)
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
    
    def flush_to_disk(self) -> None:
        """Memory cache doesn't need to flush to disk."""
        pass
    
    def _generate_key(self, key: Any) -> str:
        """Generate a cache key from the input."""
        if isinstance(key, (str, int, float, bool)):
            return str(key)
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest() 