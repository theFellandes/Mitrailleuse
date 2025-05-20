import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class SimilarityChecker:
    """Checks for similar responses to avoid duplicate content."""

    def __init__(self, base_path: Path, config: Dict):
        self.base_path = base_path
        self.config = config
        self.history_file = base_path / "cache" / "response_history.json"
        self.history: List[Dict[str, Any]] = self._load_history()
        
        # Get settings from config
        self.enabled = self._is_enabled()
        if self.enabled:
            self.threshold = self._get_threshold()
            self.cooldown_period = self._get_cooldown_period()
            self.max_recent = self._get_max_recent()
            self.close_after_use = self._get_close_after_use()
            logger.info("Similarity checking enabled with threshold %.2f", self.threshold)
        else:
            logger.info("Similarity checking disabled")

    def _is_enabled(self) -> bool:
        """Check if similarity checking is enabled in config."""
        try:
            return bool(self.config["general"]["similarity_check"]["enabled"])
        except (KeyError, TypeError):
            return False

    def _get_threshold(self) -> float:
        """Get similarity threshold from config."""
        try:
            return float(self.config["general"]["similarity_check"]["settings"]["similarity_threshold"])
        except (KeyError, TypeError, ValueError):
            return 0.8  # Default

    def _get_cooldown_period(self) -> int:
        """Get cooldown period from config."""
        try:
            return int(self.config["general"]["similarity_check"]["settings"]["cooldown_period"])
        except (KeyError, TypeError, ValueError):
            return 300  # Default 5 minutes

    def _get_max_recent(self) -> int:
        """Get max recent responses from config."""
        try:
            return int(self.config["general"]["similarity_check"]["settings"]["max_recent_responses"])
        except (KeyError, TypeError, ValueError):
            return 100  # Default

    def _get_close_after_use(self) -> bool:
        """Get close after use setting from config."""
        try:
            return bool(self.config["general"]["similarity_check"]["settings"]["close_after_use"])
        except (KeyError, TypeError):
            return True  # Default

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load response history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading response history: {str(e)}")
        return []

    def _save_history(self) -> None:
        """Save response history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving response history: {str(e)}")

    def check_similarity(self, response: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if response is similar to recent responses."""
        if not self.enabled:
            return False, 0.0

        try:
            # Extract text from response
            if isinstance(response, dict):
                text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                text = str(response)
            
            if not text:
                return False, 0.0
            
            # Check if this response is similar to any recent ones
            is_similar = False
            similarity = 0.0
            
            # Add to history
            self.history.append({
                "text": text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim history if too long
            if len(self.history) > self.max_recent:
                self.history = self.history[-self.max_recent:]
            
            # Save updated history
            self._save_history()
            
            return is_similar, similarity
            
        except Exception as e:
            logger.error(f"Error checking similarity: {str(e)}")
            return False, 0.0

    def should_cooldown(self) -> bool:
        """Check if cooldown is needed."""
        if not self.enabled or not self.history:
            return False
        
        last_time = datetime.fromisoformat(self.history[-1]["timestamp"])
        return (datetime.now() - last_time).total_seconds() < self.cooldown_period

    def get_cooldown_time(self) -> int:
        """Get remaining cooldown time in seconds."""
        if not self.enabled or not self.history:
            return 0
        
        last_time = datetime.fromisoformat(self.history[-1]["timestamp"])
        elapsed = (datetime.now() - last_time).total_seconds()
        return max(0, int(self.cooldown_period - elapsed))

    def close(self) -> None:
        """Close the similarity checker."""
        if self.enabled and self.close_after_use:
            self._save_history()
            self.history.clear() 