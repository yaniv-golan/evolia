"""Custom exceptions for the Evolia package"""

from typing import Any, Dict, Optional

class EvoliaError(Exception):
    """Base exception class for Evolia"""
    pass

class PlanGenerationError(EvoliaError):
    """Error during plan generation"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class ExecutorError(EvoliaError):
    """Error during execution"""
    def __init__(self, message: str, code: Optional[str] = None, step_name: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.step_name = step_name
        self.details = details or {}

class CodeGenerationError(EvoliaError):
    """Error during code generation"""
    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

class CodeValidationError(EvoliaError):
    """Error during code validation"""
    def __init__(self, message: str, code: str = None, validation_results: dict = None):
        super().__init__(message)
        self.code = code
        self.validation_results = validation_results or {}

class PlanValidationError(EvoliaError):
    """Error during plan validation"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class PlanExecutionError(EvoliaError):
    """Error during plan execution"""
    def __init__(self, message: str, step_name: str = None, details: dict = None):
        super().__init__(message)
        self.step_name = step_name
        self.details = details or {}

class CodeExecutionError(EvoliaError):
    """Error during code execution"""
    def __init__(self, message: str, test_results: dict = None, execution_error: str = None):
        super().__init__(message)
        self.test_results = test_results or {}
        self.execution_error = execution_error

class SecurityViolationError(EvoliaError):
    """Security violation detected"""
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.violations = self.details.get('violations', {})

class ValidationConfigError(EvoliaError):
    """Error in validation configuration"""
    def __init__(self, message: str, config: dict = None):
        super().__init__(message)
        self.config = config or {}

class TestConfigError(EvoliaError):
    """Error in test configuration"""
    def __init__(self, message: str, config: dict = None):
        super().__init__(message)
        self.config = config or {}

class FileAccessViolationError(EvoliaError):
    """File access violation detected"""
    pass

class RuntimeFixError(EvoliaError):
    """Error during runtime error fixing"""
    def __init__(self, message: str, code: str = None, fix_attempts: list = None):
        super().__init__(message)
        self.code = code
        self.fix_attempts = fix_attempts or []

class SyntaxFixError(ExecutorError):
    """Error during syntax fixing"""
    pass

class FileAccessError(ExecutorError):
    """Error during file access"""
    pass

class ValidationError(ExecutorError):
    """Error during validation"""
    pass

class SecurityError(ExecutorError):
    """Error during security checks"""
    pass