"""Unit tests for interface verification."""

from typing import Any, Dict

import pytest

from evolia.core.interface_verification import verify_interface
from evolia.models.models import FunctionInterface, Parameter


@pytest.fixture
def test_interface():
    """Create a test interface for validation."""
    return FunctionInterface(
        function_name="add_numbers",
        parameters=[
            Parameter(name="a", type="int", description="First number"),
            Parameter(name="b", type="int", description="Second number"),
        ],
        return_type="int",
        description="Add two numbers",
    )


@pytest.fixture
def valid_generated_code():
    """Create valid generated code for testing."""
    return {
        "code": "def add_numbers(a: int, b: int) -> int: return a + b",
        "function_info": {
            "name": "add_numbers",
            "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "int",
            "docstring": "Add two numbers",
        },
    }


def test_verify_interface_valid_outputs(test_interface, valid_generated_code):
    """Test verification with valid outputs."""
    errors = verify_interface(test_interface, valid_generated_code)
    assert not errors, f"Expected no errors but got: {errors}"


def test_verify_interface_missing_output(test_interface):
    """Test verification with missing output."""
    generated_code = {
        "code": "def add_numbers(a: int, b: int) -> None: pass",
        "function_info": {
            "name": "add_numbers",
            "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "None",
            "docstring": "Add two numbers",
        },
    }

    errors = verify_interface(test_interface, generated_code)
    assert any("Return type mismatch" in error for error in errors)


def test_verify_interface_wrong_output_type(test_interface):
    """Test verification with wrong output type."""
    generated_code = {
        "code": "def add_numbers(a: int, b: int) -> str: return str(a + b)",
        "function_info": {
            "name": "add_numbers",
            "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "str",
            "docstring": "Add two numbers",
        },
    }

    errors = verify_interface(test_interface, generated_code)
    assert any("Return type mismatch" in error for error in errors)


def test_verify_interface_invalid_reference_format(test_interface):
    """Test verification with invalid reference format."""
    generated_code = {
        "code": "def wrong_name(a: int, b: int) -> int: return a + b",
        "function_info": {
            "name": "wrong_name",
            "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "int",
            "docstring": "Add two numbers",
        },
    }

    errors = verify_interface(test_interface, generated_code)
    assert any("Function name mismatch" in error for error in errors)


def test_verify_interface_multiple_outputs(test_interface, valid_generated_code):
    """Test verification with multiple outputs."""
    errors = verify_interface(test_interface, valid_generated_code)
    assert not errors, f"Expected no errors but got: {errors}"
