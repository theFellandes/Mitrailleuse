import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of API responses."""

    def __init__(self, base_path: Path, config: Dict):
        self.base_path = base_path
        self.config = config
        self.cache_dir = base_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "responses.json"
        self.cache: Dict[str, Any] = self._load_cache()
        
        # Set temp directory to task's cache directory
        os.environ['TMPDIR'] = str(self.cache_dir)
        tempfile.tempdir = str(self.cache_dir)

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
        return {}

    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def get(self, key: str, service: str) -> Optional[Any]:
        """Get a value from cache."""
        cache_key = f"{service}_{key}"
        return self.cache.get(cache_key)

    def set(self, key: str, value: Any, service: str) -> None:
        """Set a value in cache."""
        cache_key = f"{service}_{key}"
        self.cache[cache_key] = value
        self._save_cache()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "total_items": len(self.cache),
            "services": len(set(k.split('_')[0] for k in self.cache.keys()))
        }

    def clear_all(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self._save_cache()

    def close(self) -> None:
        """Close the cache manager."""
        self._save_cache() 