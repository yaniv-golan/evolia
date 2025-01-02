"""Integration tests for code execution functionality."""
import pytest
from evolia.core.code_validation import execute_test_cases

def test_execute_basic_test_cases():
    """Test execution of basic test cases."""
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
    """Test execution of failing test cases."""
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
    """Test execution with timeout."""
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
    """Test execution with runtime error."""
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

def test_execute_with_imports():
    """Test execution with allowed imports."""
    code = """
import math
def calculate_circle_area(radius: float) -> float:
    return math.pi * radius ** 2
"""
    test_cases = [
        {'inputs': [1], 'expected': math.pi},
        {'inputs': [2], 'expected': 4 * math.pi}
    ]
    
    results = execute_test_cases(code, test_cases)
    assert results['passed'] == 2
    assert results['failed'] == 0

def test_execute_with_type_validation():
    """Test execution with type validation."""
    code = """
def process_list(items: list[int]) -> int:
    return sum(items)
"""
    test_cases = [
        {'inputs': [[1, 2, 3]], 'expected': 6},
        {'inputs': [["not", "integers"]], 'expected': None}  # Should fail type check
    ]
    
    results = execute_test_cases(code, test_cases)
    assert results['passed'] == 1
    assert results['failed'] == 1
    assert any('type' in str(error).lower() for error in results['failures']) 