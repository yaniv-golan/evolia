"""Tests for custom exceptions"""
import pytest

from evolia.utils.exceptions import (
    CodeExecutionError,
    CodeGenerationError,
    CodeValidationError,
    EvoliaError,
    ExecutorError,
    FileAccessViolationError,
    PlanExecutionError,
    PlanGenerationError,
    PlanValidationError,
    RuntimeFixError,
    SecurityViolationError,
    SyntaxFixError,
    TestConfigError,
    ValidationConfigError,
)


def test_code_generation_error():
    """Test CodeGenerationError creation and attributes"""
    error = CodeGenerationError(
        message="Failed to generate code",
        code="def test(): pass",
        details={"error": "syntax error"},
    )

    assert str(error) == "Failed to generate code"
    assert error.code == "def test(): pass"
    assert error.details == {"error": "syntax error"}


def test_code_validation_error():
    """Test code validation error message."""
    error = CodeValidationError("Invalid syntax, Wrong return type")
    assert "Invalid syntax, Wrong return type" in str(error)


def test_code_execution_error():
    """Test code execution error message."""
    error = CodeExecutionError("Division by zero")
    assert "Division by zero" in str(error)


def test_security_violation_error():
    """Test security violation error message."""
    error = SecurityViolationError("Unauthorized import detected")
    assert "Unauthorized import detected" in str(error)


def test_validation_config_error():
    """Test validation config error message."""
    error = ValidationConfigError("Invalid configuration")
    assert "Invalid configuration" in str(error)


def test_test_config_error():
    """Test test config error message."""
    error = TestConfigError("Invalid test configuration")
    assert "Invalid test configuration" in str(error)


def test_error_inheritance():
    """Test error class inheritance."""
    error = SecurityViolationError("test")
    assert isinstance(error, EvoliaError)
    assert isinstance(error, Exception)

    error = CodeValidationError("test")
    assert isinstance(error, CodeValidationError)
    assert isinstance(error, EvoliaError)
    assert isinstance(error, Exception)
