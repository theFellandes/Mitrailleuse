import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from .logger import get_logger

class CacheManager:
    """Manages both in-memory and file-based caching for API responses."""

    def __init__(self, base_path: Path, config: Dict[str, Any], cache_name: str = "cache"):
        """
        Initialize the cache manager.
        
        Args:
            base_path: Base path where cache files will be stored
            config: Configuration dictionary
            cache_name: Name of the cache directory
        """
        self.base_path = base_path
        self.cache_dir = base_path / cache_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger("cache_manager", config)
        
        # Initialize in-memory cache
        self._memory_cache: Dict[str, Any] = {}
        self._memory_cache_hits = 0
        self._memory_cache_misses = 0
        
        # Initialize file cache stats
        self._file_cache_hits = 0
        self._file_cache_misses = 0
        
        # Load existing cache stats if available
        self._load_cache_stats()
        self.logger.info("Cache manager initialized", {"cache_dir": str(self.cache_dir)})

    def _compute_cache_key(self, data: Dict) -> str:
        """
        Compute a deterministic hash for caching.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """
        Get the path for a cache file.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache_stats(self) -> None:
        """Load cache statistics from disk."""
        stats_file = self.cache_dir / "cache_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    self._memory_cache_hits = stats.get("memory_hits", 0)
                    self._memory_cache_misses = stats.get("memory_misses", 0)
                    self._file_cache_hits = stats.get("file_hits", 0)
                    self._file_cache_misses = stats.get("file_misses", 0)
                self.logger.debug("Cache stats loaded from disk")
            except Exception as e:
                self.logger.error(f"Error loading cache stats: {str(e)}")

    def _save_cache_stats(self) -> None:
        """Save cache statistics to disk."""
        stats = {
            "memory_hits": self._memory_cache_hits,
            "memory_misses": self._memory_cache_misses,
            "file_hits": self._file_cache_hits,
            "file_misses": self._file_cache_misses,
            "last_updated": datetime.now().isoformat()
        }
        stats_file = self.cache_dir / "cache_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            self.logger.debug("Cache stats saved to disk")
        except Exception as e:
            self.logger.error(f"Error saving cache stats: {str(e)}")

    def get(self, data: Dict, service: str = "openai") -> Optional[Any]:
        """
        Get data from cache (tries memory first, then file).
        
        Args:
            data: Data to look up
            service: Service name (e.g., "openai", "deepl")
            
        Returns:
            Cached data if found, None otherwise
        """
        cache_key = f"{service}_{self._compute_cache_key(data)}"
        
        # Try memory cache first
        if cache_key in self._memory_cache:
            self._memory_cache_hits += 1
            self.logger.debug(f"Memory cache hit for {service} request")
            return self._memory_cache[cache_key]
        self._memory_cache_misses += 1
        
        # Try file cache
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                self._file_cache_hits += 1
                self.logger.debug(f"File cache hit for {service} request")
                # Also update memory cache
                self._memory_cache[cache_key] = cached_data
                return cached_data
            except Exception as e:
                self.logger.error(f"Error reading from file cache: {str(e)}")
        
        self._file_cache_misses += 1
        return None

    def set(self, data: Dict, response: Any, service: str = "openai") -> None:
        """
        Save data to both memory and file cache.
        
        Args:
            data: Input data
            response: Response to cache
            service: Service name (e.g., "openai", "deepl")
        """
        cache_key = f"{service}_{self._compute_cache_key(data)}"
        
        # Save to memory cache
        self._memory_cache[cache_key] = response
        
        # Save to file cache
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f, indent=2)
            self.logger.debug(f"Saved response to cache for {service}")
        except Exception as e:
            self.logger.error(f"Error writing to file cache: {str(e)}")

    def clear_memory_cache(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()
        self.logger.info("Memory cache cleared")

    def clear_file_cache(self) -> None:
        """Clear the file cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "cache_stats.json":
                    cache_file.unlink()
            self.logger.info("File cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing file cache: {str(e)}")

    def clear_all(self) -> None:
        """Clear both memory and file cache."""
        self.clear_memory_cache()
        self.clear_file_cache()

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Save stats before returning
        self._save_cache_stats()
        
        stats = {
            "memory_hits": self._memory_cache_hits,
            "memory_misses": self._memory_cache_misses,
            "memory_size": len(self._memory_cache),
            "file_hits": self._file_cache_hits,
            "file_misses": self._file_cache_misses,
            "file_size": len(list(self.cache_dir.glob("*.json"))) - 1  # Exclude stats file
        }
        
        self.logger.debug("Cache stats retrieved", {"stats": stats})
        return stats

    def close(self) -> None:
        """Close the logger."""
        self.logger.close() 