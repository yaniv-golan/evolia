"""Tests for custom exceptions"""
import pytest
from evolia.utils.exceptions import (
    EvoliaError,
    ExecutorError,
    CodeGenerationError,
    CodeValidationError,
    PlanValidationError,
    PlanExecutionError,
    CodeExecutionError,
    SecurityViolationError,
    ValidationConfigError,
    TestConfigError,
    FileAccessViolationError,
    RuntimeFixError,
    SyntaxFixError,
    PlanGenerationError
)

def test_code_generation_error():
    """Test CodeGenerationError creation and attributes"""
    error = CodeGenerationError(
        message="Failed to generate code",
        code="def test(): pass",
        details={'error': 'syntax error'}
    )
    
    assert str(error) == "Failed to generate code"
    assert error.code == "def test(): pass"
    assert error.details == {'error': 'syntax error'}

def test_code_validation_error():
    """Test CodeValidationError creation and string representation"""
    error = CodeValidationError(
        message="Validation failed",
        validation_results={
            'issues': ['Invalid syntax', 'Wrong return type']
        }
    )
    
    assert "Validation failed: Invalid syntax, Wrong return type" in str(error)
    assert error.validation_results['issues'] == ['Invalid syntax', 'Wrong return type']

def test_code_execution_error():
    """Test CodeExecutionError creation and string representation"""
    execution_error = ValueError("Division by zero")
    error = CodeExecutionError(
        message="Test execution failed",
        test_results={'failed': 1},
        execution_error=execution_error
    )
    
    assert "Test execution failed: Division by zero" in str(error)
    assert error.test_results == {'failed': 1}
    assert error.execution_error == execution_error

def test_security_violation_error():
    """Test SecurityViolationError creation and string representation"""
    error = SecurityViolationError(
        message="Security check failed",
        violations={
            'system_calls': ['os.system() call detected'],
            'file_operations': ['open() call detected']
        },
        code="import os\nos.system('rm -rf /')"
    )
    
    assert "Security check failed" in str(error)
    assert "system_calls: os.system() call detected" in str(error)
    assert "file_operations: open() call detected" in str(error)
    assert error.code == "import os\nos.system('rm -rf /')"

def test_validation_config_error():
    """Test ValidationConfigError creation and string representation"""
    error = ValidationConfigError(
        message="Missing required field 'security_checks'",
        section="validation"
    )
    
    assert "Invalid configuration in validation" in str(error)
    assert "Missing required field 'security_checks'" in str(error)
    assert error.section == "validation"

def test_test_config_error():
    """Test TestConfigError creation and string representation"""
    test_case = {
        'inputs': [1, 2],
        'expected': 'not a number'
    }
    error = TestConfigError(
        message="Invalid test case: expected result must be a number",
        case=test_case
    )
    
    assert "Invalid test case: expected result must be a number" in str(error)
    assert error.case == test_case

def test_error_inheritance():
    """Test exception inheritance hierarchy"""
    error = CodeGenerationError("test")
    assert isinstance(error, CodeGenerationError)
    assert isinstance(error, EvoliaError)
    assert isinstance(error, Exception)
    
    error = SecurityViolationError("test")
    assert isinstance(error, SecurityViolationError)
    assert isinstance(error, CodeValidationError)
    assert isinstance(error, EvoliaError)
    assert isinstance(error, Exception) 