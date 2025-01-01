"""Logging configuration for Evolia"""
import logging
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from ..models.models import Parameter, OutputDefinition, CodeGenerationRequest, CodeResponse

class ModelJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for model classes."""
    def default(self, obj):
        if isinstance(obj, Parameter):
            return {
                'name': obj.name,
                'type': obj.type,
                'description': obj.description
            }
        elif isinstance(obj, OutputDefinition):
            return {
                'type': obj.type,
                'description': getattr(obj, 'description', None)
            }
        elif isinstance(obj, (CodeGenerationRequest, CodeResponse)):
            return obj.__dict__
        return super().default(obj)

class LogConfig:
    """Configuration for logging setup"""
    def __init__(self, 
                 verbose: bool = False,
                 log_file: Optional[Path] = None,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG):
        self.verbose = verbose
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level

class LogHandler:
    """Handler for log messages"""
    def __init__(self, config: LogConfig):
        self.config = config
        self.handlers = []
        
    def setup_console_handler(self):
        """Set up console logging handler"""
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(self.config.console_level)
        console.setFormatter(ColorFormatter('%(levelname)s - %(message)s'))
        console.addFilter(ConsoleFilter(self.config.verbose))
        self.handlers.append(console)
        
    def setup_file_handler(self):
        """Set up file logging handler if configured"""
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(self.config.file_level)
            file_handler.setFormatter(LogFormatter('%(asctime)s [%(levelname)8s] %(message)s'))
            self.handlers.append(file_handler)

class LogFormatter(logging.Formatter):
    """Base formatter for log messages"""
    def format(self, record):
        if hasattr(record, 'payload') and isinstance(record.payload, (dict, list)):
            record.msg = f"{record.msg}\nPayload: {json.dumps(record.payload, indent=2, cls=ModelJSONEncoder)}"
        return super().format(record)

class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels and structured logging"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',   # Green
        'WARNING': '\033[0;33m', # Yellow
        'ERROR': '\033[0;31m',   # Red
        'CRITICAL': '\033[0;35m' # Purple
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        # Add color to the level name if colors are enabled
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        # Format extra fields for structured logging
        if hasattr(record, 'payload') and isinstance(record.payload, (dict, list)):
            record.msg = f"{record.msg}\nPayload: {json.dumps(record.payload, indent=2, cls=ModelJSONEncoder)}"
        
        return super().format(record)

class ConsoleFilter(logging.Filter):
    """Filter to control which messages go to console based on verbose mode"""
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        
    def filter(self, record):
        # In non-verbose mode, only show high-level progress messages
        if not self.verbose:
            return (
                not hasattr(record, 'payload') and  # No structured logging
                record.levelno >= logging.INFO and  # Only INFO and above
                not record.name.startswith('evolia.')  # No internal module logs
            )
        return True

def setup_logger(log_file: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """Set up and return a logger with both console and file handlers
    
    Args:
        log_file: Optional path to log file. If not provided, will use 'evolia.log'
                 in the current directory.
        verbose: Whether to show detailed logs in console output
    """
    logger = logging.getLogger('evolia')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG by default for detailed logging
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with filtering based on verbose mode
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(ConsoleFilter(verbose))
    if verbose:
        console_handler.setLevel(logging.DEBUG)
        console_format = '%(asctime)s - %(levelname)s - %(message)s'
    else:
        console_handler.setLevel(logging.INFO)
        console_format = '%(message)s'  # Clean output in non-verbose mode
    console_handler.setFormatter(ColorFormatter(console_format))
    logger.addHandler(console_handler)
    
    # File handler always gets everything
    if log_file is None:
        log_file = Path('output.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_handler.setFormatter(logging.Formatter(file_format))
    logger.addHandler(file_handler)
    
    return logger

@contextmanager
def code_generation_context(step_name: str):
    """Context manager for code generation steps."""
    logger = logging.getLogger('evolia')
    logger.info(f"Starting code generation: {step_name}")
    try:
        yield
        logger.info(f"Completed code generation: {step_name}")
    except Exception as e:
        logger.error(f"Error in code generation: {step_name}", exc_info=True)
        raise

@contextmanager
def validation_context(step_name: str):
    """Context manager for validation steps."""
    logger = logging.getLogger('evolia')
    logger.info(f"Starting validation: {step_name}")
    try:
        yield
        logger.info(f"Completed validation: {step_name}")
    except Exception as e:
        logger.error(f"Error in validation: {step_name}", exc_info=True)
        raise

@contextmanager
def execution_context(step_name: str):
    """Context manager for execution steps."""
    logger = logging.getLogger('evolia')
    logger.info(f"Starting execution: {step_name}")
    try:
        yield
        logger.info(f"Completed execution: {step_name}")
    except Exception as e:
        logger.error(f"Error in execution: {step_name}", exc_info=True)
        raise

class CustomFormatter(logging.Formatter):
    """Custom formatter that includes payload data as JSON."""
    def format(self, record):
        if not hasattr(record, 'payload'):
            record.payload = {}
        
        # Format the message first
        record.msg = f"{record.msg}\nPayload: {json.dumps(record.payload, indent=2, cls=ModelJSONEncoder)}"
        
        # Call the parent formatter
        return super().format(record)