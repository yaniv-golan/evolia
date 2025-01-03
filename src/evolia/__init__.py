"""
A task orchestration system using OpenAI for code generation and execution.
"""

__version__ = "0.1.0"

from .utils.exceptions import (
    CodeExecutionError,
    CodeGenerationError,
    CodeValidationError,
    TestConfigError,
    ValidationConfigError,
)
