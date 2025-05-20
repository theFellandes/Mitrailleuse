import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from datasketch import MinHash, MinHashLSH
import jieba
from .logger import get_logger

class SimilarityChecker:
    """Utility class for checking response similarity using MinHash."""

    def __init__(self, base_path: Path, config: Dict[str, Any]):
        """
        Initialize the similarity checker.
        
        Args:
            base_path: Base path where files will be stored
            config: Configuration dictionary
        """
        self.base_path = base_path
        self.config = config
        self.logger = get_logger("similarity_checker", config)
        
        # Initialize LSH index
        self.lsh = MinHashLSH(threshold=0.5, num_perm=128)
        
        # Store recent responses and their timestamps
        self.recent_responses: List[Tuple[MinHash, datetime]] = []
        
        # Get configuration
        self.similarity_threshold = self.config.get("general", {}).get("similarity_threshold", 0.8)
        self.cooldown_period = self.config.get("general", {}).get("cooldown_period", 300)  # 5 minutes default
        self.max_recent_responses = self.config.get("general", {}).get("max_recent_responses", 100)
        
        # Load existing responses if available
        self._load_recent_responses()

    def _load_recent_responses(self) -> None:
        """Load recent responses from disk."""
        history_file = self.base_path / "cache" / "response_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    for item in history:
                        minhash = MinHash(num_perm=128)
                        minhash.update(item["tokens"].encode())
                        self.recent_responses.append((
                            minhash,
                            datetime.fromisoformat(item["timestamp"])
                        ))
                self.logger.info(f"Loaded {len(self.recent_responses)} recent responses")
            except Exception as e:
                self.logger.error(f"Error loading response history: {str(e)}")

    def _save_recent_responses(self) -> None:
        """Save recent responses to disk."""
        history_file = self.base_path / "cache" / "response_history.json"
        try:
            history = [
                {
                    "tokens": " ".join(minhash.digest()),
                    "timestamp": timestamp.isoformat()
                }
                for minhash, timestamp in self.recent_responses
            ]
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            self.logger.debug("Saved response history to disk")
        except Exception as e:
            self.logger.error(f"Error saving response history: {str(e)}")

    def _create_minhash(self, text: str) -> MinHash:
        """
        Create a MinHash from text.
        
        Args:
            text: Text to hash
            
        Returns:
            MinHash object
        """
        # Tokenize text (using jieba for Chinese text)
        tokens = list(jieba.cut(text))
        
        # Create MinHash
        minhash = MinHash(num_perm=128)
        for token in tokens:
            minhash.update(token.encode())
        
        return minhash

    def _cleanup_old_responses(self) -> None:
        """Remove responses older than cooldown period."""
        now = datetime.now()
        self.recent_responses = [
            (minhash, timestamp)
            for minhash, timestamp in self.recent_responses
            if now - timestamp < timedelta(seconds=self.cooldown_period)
        ]
        
        # Limit the number of stored responses
        if len(self.recent_responses) > self.max_recent_responses:
            self.recent_responses = self.recent_responses[-self.max_recent_responses:]

    def check_similarity(self, response: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if response is similar to recent responses.
        
        Args:
            response: Response to check
            
        Returns:
            Tuple of (is_similar, similarity_score)
        """
        # Extract text from response
        if "choices" in response:
            # OpenAI/DeepSeek format
            text = response["choices"][0]["message"]["content"]
        elif "translated_text" in response:
            # DeepL format
            text = response["translated_text"]
        else:
            self.logger.warning("Unknown response format")
            return False, 0.0
        
        # Create MinHash for response
        minhash = self._create_minhash(text)
        
        # Clean up old responses
        self._cleanup_old_responses()
        
        # Check similarity with recent responses
        max_similarity = 0.0
        for recent_minhash, _ in self.recent_responses:
            similarity = minhash.jaccard(recent_minhash)
            max_similarity = max(max_similarity, similarity)
        
        # Add to recent responses
        self.recent_responses.append((minhash, datetime.now()))
        self._save_recent_responses()
        
        is_similar = max_similarity >= self.similarity_threshold
        if is_similar:
            self.logger.warning(
                f"Similar response detected (similarity: {max_similarity:.2f})",
                {"similarity": max_similarity}
            )
        
        return is_similar, max_similarity

    def should_cooldown(self) -> bool:
        """
        Check if we should apply cooldown based on recent responses.
        
        Returns:
            True if cooldown should be applied
        """
        now = datetime.now()
        recent_count = sum(
            1 for _, timestamp in self.recent_responses
            if now - timestamp < timedelta(seconds=self.cooldown_period)
        )
        
        return recent_count >= self.max_recent_responses

    def get_cooldown_time(self) -> int:
        """
        Get remaining cooldown time in seconds.
        
        Returns:
            Remaining cooldown time
        """
        if not self.recent_responses:
            return 0
        
        now = datetime.now()
        oldest_timestamp = min(timestamp for _, timestamp in self.recent_responses)
        elapsed = (now - oldest_timestamp).total_seconds()
        
        return max(0, int(self.cooldown_period - elapsed))

    def close(self) -> None:
        """Close the logger."""
        self.logger.close() 