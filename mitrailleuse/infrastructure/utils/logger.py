import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extras import Json

class Logger:
    """A comprehensive logging system that supports both file and database logging."""

    def __init__(
        self,
        name: str,
        log_dir: Path,
        config: Dict[str, Any],
        db_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            config: Logging configuration
            db_config: Optional database configuration
        """
        self.name = name
        self.log_dir = log_dir
        self.config = config
        self.db_config = db_config
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Add console handler
        self._add_console_handler()
        
        # Add file handler
        self._add_file_handler()
        
        # Initialize database connection if configured
        self.db_conn = None
        if self.db_config and self.db_config.get("enabled", False):
            self._init_db_connection()

    def _add_console_handler(self) -> None:
        """Add console handler to logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self) -> None:
        """Add file handler to logger."""
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _init_db_connection(self) -> None:
        """Initialize database connection."""
        try:
            self.db_conn = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["username"],
                password=self.db_config["password"]
            )
            self._create_log_table()
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            self.db_conn = None

    def _create_log_table(self) -> None:
        """Create log table if it doesn't exist."""
        if not self.db_conn:
            return

        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE,
                        logger_name VARCHAR(255),
                        level VARCHAR(50),
                        message TEXT,
                        module VARCHAR(255),
                        function VARCHAR(255),
                        line_number INTEGER,
                        extra_data JSONB
                    )
                """)
            self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to create log table: {str(e)}")

    def _log_to_db(
        self,
        level: str,
        message: str,
        extra_data: Optional[Dict] = None
    ) -> None:
        """
        Log message to database.
        
        Args:
            level: Log level
            message: Log message
            extra_data: Additional data to log
        """
        if not self.db_conn:
            return

        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO logs (
                        timestamp,
                        logger_name,
                        level,
                        message,
                        module,
                        function,
                        line_number,
                        extra_data
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now(),
                    self.name,
                    level,
                    message,
                    extra_data.get("module", ""),
                    extra_data.get("function", ""),
                    extra_data.get("line_number", 0),
                    Json(extra_data) if extra_data else None
                ))
            self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to log to database: {str(e)}")

    def _get_extra_data(self) -> Dict[str, Any]:
        """
        Get extra data for logging.
        
        Returns:
            Dictionary with extra data
        """
        import inspect
        frame = inspect.currentframe().f_back
        return {
            "module": frame.f_globals.get("__name__", ""),
            "function": frame.f_code.co_name,
            "line_number": frame.f_lineno
        }

    def debug(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log debug message."""
        self.logger.debug(message)
        if self.db_conn:
            self._log_to_db("DEBUG", message, extra_data or self._get_extra_data())

    def info(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log info message."""
        self.logger.info(message)
        if self.db_conn:
            self._log_to_db("INFO", message, extra_data or self._get_extra_data())

    def warning(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log warning message."""
        self.logger.warning(message)
        if self.db_conn:
            self._log_to_db("WARNING", message, extra_data or self._get_extra_data())

    def error(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log error message."""
        self.logger.error(message)
        if self.db_conn:
            self._log_to_db("ERROR", message, extra_data or self._get_extra_data())

    def critical(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log critical message."""
        self.logger.critical(message)
        if self.db_conn:
            self._log_to_db("CRITICAL", message, extra_data or self._get_extra_data())

    def close(self) -> None:
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None

def get_logger(name: str, config: Dict[str, Any]) -> Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    log_dir = Path(config.get("general", {}).get("logs", {}).get("log_dir", "logs"))
    db_config = config.get("general", {}).get("db", {})
    
    return Logger(name, log_dir, config, db_config) 