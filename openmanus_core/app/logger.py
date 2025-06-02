import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

_print_level = "INFO"
_file_logging_configured = False

# Configure a basic console logger first
_logger.remove()
_logger.add(sys.stderr, level=_print_level)

def setup_file_logging(project_root_path: Path, name: Optional[str] = None, logfile_level="DEBUG"):
    global _file_logging_configured
    if _file_logging_configured:
        return

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )
    
    logs_dir = project_root_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists
    
    _logger.add(logs_dir / f"{log_name}.log", level=logfile_level)
    _file_logging_configured = True
    _logger.info(f"File logging configured at: {logs_dir / f'{log_name}.log'}")

# Export the logger instance directly
logger = _logger


if __name__ == "__main__":
    # For testing logger.py itself
    test_project_root = Path(__file__).resolve().parent.parent # Simulates openmanus_core/
    setup_file_logging(test_project_root, name="test_log")
    
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
