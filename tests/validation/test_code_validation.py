"""Tests for code validation functionality"""
import pytest
from evolia.validation.code_validation import validate_python_code, execute_test_cases, ValidationResult
from evolia.utils.exceptions import CodeValidationError

def test_validate_basic_function():
    """Test validation of a basic function"""
    code = """
def add(a, b):
    return a + b
"""
    requirements = {
        'function_name': 'add',
        'parameters': ['a', 'b']
    }
    
    result = validate_python_code(code, requirements)
    assert result.is_valid
    assert not result.issues
    assert result.details['function_name'] == 'add'
    assert result.details['parameters'] == ['a', 'b']

def test_validate_function_with_type_hints():
    """Test validation of a function with type hints"""
    code = """
def multiply(x: int, y: int) -> int:
    return x * y
"""
    requirements = {
        'function_name': 'multiply',
        'parameters': ['x', 'y'],
        'return_type': 'int'
    }
    
    result = validate_python_code(code, requirements)
    assert result.is_valid
    assert not result.issues
    assert result.details['return_type'] == 'int'

def test_validate_function_name_mismatch():
    """Test validation when function name doesn't match"""
    code = """
def wrong_name(x, y):
    return x + y
"""
    requirements = {
        'function_name': 'add'
    }
    
    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any('name mismatch' in issue.lower() for issue in result.issues)

def test_validate_parameter_mismatch():
    """Test validation when parameters don't match"""
    code = """
def func(wrong1, wrong2):
    return wrong1 + wrong2
"""
    requirements = {
        'parameters': ['a', 'b']
    }
    
    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any('parameter name mismatch' in issue.lower() for issue in result.issues)

def test_validate_return_type_mismatch():
    """Test validation when return type doesn't match"""
    code = """
def func(x: int, y: int) -> float:
    return x + y
"""
    requirements = {
        'return_type': 'int'
    }
    
    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any('return type mismatch' in issue.lower() for issue in result.issues)

def test_validate_syntax_error():
    """Test validation of code with syntax error"""
    code = """
def bad_syntax(x, y)
    return x + y
"""
    result = validate_python_code(code, {})
    assert not result.is_valid
    assert any('syntax error' in issue.lower() for issue in result.issues)
    assert not result.details['syntax_valid']

def test_validate_no_function():
    """Test validation of code without a function"""
    code = """
x = 1
y = 2
result = x + y
"""
    result = validate_python_code(code, {})
    assert not result.is_valid
    assert any('no function' in issue.lower() for issue in result.issues)
    assert not result.details['has_function']

def test_validate_constraints():
    """Test validation with constraints"""
    code = """
global_var = 0

def func_with_global():
    global global_var
    return global_var + 1

def nested():
    def inner():
        return 0
    return inner()
"""
    requirements = {
        'constraints': ['no_globals', 'no_nested_functions']
    }
    
    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any('global variables' in issue.lower() for issue in result.issues)
    assert any('nested function' in issue.lower() for issue in result.issues)

def test_execute_basic_test_cases():
    """Test execution of basic test cases"""
    code = """
def add(a, b):
    return a + b
"""
    test_cases = [
        {'inputs': [1, 2], 'expected': 3},
        {'inputs': [-1, 1], 'expected': 0},
        {'inputs': [0, 0], 'expected': 0}
    ]
    
    results = execute_test_cases(code, test_cases)
    assert results['passed'] == 3
    assert results['failed'] == 0
    assert not results['failures']

def test_execute_failing_test_cases():
    """Test execution of failing test cases"""
    code = """
def subtract(a, b):
    return a + b  # Wrong operation
"""
    test_cases = [
        {'inputs': [3, 2], 'expected': 1},
        {'inputs': [1, 1], 'expected': 0}
    ]
    
    results = execute_test_cases(code, test_cases)
    assert results['passed'] == 0
    assert results['failed'] == 2
    assert len(results['failures']) == 2

def test_execute_timeout():
    """Test execution with timeout"""
    code = """
def infinite_loop(x):
    while True:
        x += 1
    return x
"""
    test_cases = [
        {'inputs': [1], 'expected': 2}
    ]
    
    results = execute_test_cases(code, test_cases, timeout=1)
    assert results['failed'] == 1
    assert 'timeout' in results['failures'][0]['error'].lower()

def test_execute_runtime_error():
    """Test execution with runtime error"""
    code = """
def divide(a, b):
    return a / b
"""
    test_cases = [
        {'inputs': [1, 0], 'expected': None}
    ]
    
    results = execute_test_cases(code, test_cases)
    assert results['failed'] == 1
    assert 'error' in results['failures'][0] 