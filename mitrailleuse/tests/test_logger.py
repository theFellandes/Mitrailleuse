from mitrailleuse.infrastructure.logging.logger import get_logger

def test_get_logger():
    logger = get_logger("test_logger")
    logger.info("Test log message")
    assert logger is not None
