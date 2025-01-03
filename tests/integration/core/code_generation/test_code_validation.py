"""Tests for code validation functionality."""
import pytest

from evolia.utils.exceptions import CodeValidationError
from evolia.validation.code_validation import validate_python_code


def test_validate_basic_function():
    """Test validation of a basic function."""
    code = """
def add(a, b):
    return a + b
"""
    requirements = {"function_name": "add", "parameters": ["a", "b"]}

    result = validate_python_code(code, requirements)
    assert result.is_valid
    assert not result.issues
    assert result.details["function_name"] == "add"
    assert result.details["parameters"] == ["a", "b"]


def test_validate_function_with_type_hints():
    """Test validation of a function with type hints."""
    code = """
def multiply(x: int, y: int) -> int:
    return x * y
"""
    requirements = {
        "function_name": "multiply",
        "parameters": ["x", "y"],
        "return_type": "int",
    }

    result = validate_python_code(code, requirements)
    assert result.is_valid
    assert not result.issues
    assert result.details["return_type"] == "int"


def test_validate_security_imports():
    """Test validation of security-sensitive imports."""
    code = """
import os
import sys
def dangerous_func():
    os.system('rm -rf /')
"""
    requirements = {"constraints": ["no_system_calls"]}

    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any("system call" in issue.lower() for issue in result.issues)


def test_validate_file_operations():
    """Test validation of file operations."""
    code = """
def write_file(content):
    with open('/etc/passwd', 'w') as f:
        f.write(content)
"""
    requirements = {"allowed_write_paths": ["/tmp"]}

    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any("file access" in issue.lower() for issue in result.issues)


def test_validate_network_access():
    """Test validation of network access."""
    code = """
import socket
def connect():
    s = socket.socket()
    s.connect(('evil.com', 80))
"""
    requirements = {"constraints": ["no_network"]}

    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any("network" in issue.lower() for issue in result.issues)


def test_validate_code_injection():
    """Test validation of code injection attempts."""
    code = """
def evil_eval(user_input):
    return eval(user_input)
"""
    requirements = {"constraints": ["no_eval"]}

    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any("eval" in issue.lower() for issue in result.issues)


def test_validate_subprocess():
    """Test validation of subprocess usage."""
    code = """
import subprocess
def run_command(cmd):
    subprocess.run(cmd, shell=True)
"""
    requirements = {"constraints": ["no_subprocess"]}

    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any("subprocess" in issue.lower() for issue in result.issues)


def test_validate_allowed_modules():
    """Test validation of allowed modules."""
    code = """
import math
import json
import requests  # Not allowed
def process_data(data):
    return json.dumps(data)
"""
    requirements = {"allowed_modules": {"math", "json"}}

    result = validate_python_code(code, requirements)
    assert not result.is_valid
    assert any("module not allowed" in issue.lower() for issue in result.issues)


def test_validate_syntax_error():
    """Test validation of code with syntax error."""
    code = """
def bad_syntax(x, y)
    return x + y
"""
    result = validate_python_code(code, {})
    assert not result.is_valid
    assert any("syntax error" in issue.lower() for issue in result.issues)
    assert not result.details["syntax_valid"]


def test_validate_no_function():
    """Test validation of code without a function."""
    code = """
x = 1
y = 2
result = x + y
"""
    result = validate_python_code(code, {})
    assert not result.is_valid
    assert any("no function" in issue.lower() for issue in result.issues)
    assert not result.details["has_function"]
