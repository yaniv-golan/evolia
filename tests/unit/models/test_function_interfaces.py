"""Unit tests for function interfaces."""

from typing import Any, Dict

import pytest

from evolia.core.interface_verification import verify_interface
from evolia.models.models import FunctionInterface, Parameter


def test_function_parameter_validation():
    """Test parameter validation."""
    # Valid parameter
    param = Parameter(name="input_path", type="str", description="Input file path")
    assert param.name == "input_path"
    assert param.type == "str"
    assert param.description == "Input file path"

    # Invalid parameter name
    with pytest.raises(ValueError, match="Invalid parameter name"):
        Parameter(name="2invalid", type="str", description="Invalid name")

    # Missing required fields
    with pytest.raises(TypeError):
        Parameter(name="test")


def test_function_interface_validation():
    """Test function interface validation."""
    # Valid interface
    interface = FunctionInterface(
        function_name="process_file",
        parameters=[
            Parameter(name="input_path", type="str", description="Input file path"),
            Parameter(name="output_path", type="str", description="Output file path"),
        ],
        return_type="bool",
        description="Process a file and save results",
    )

    assert interface.function_name == "process_file"
    assert len(interface.parameters) == 2
    assert interface.return_type == "bool"
    assert interface.description == "Process a file and save results"

    # Missing required fields
    with pytest.raises(TypeError):
        FunctionInterface(function_name="test")


def test_verify_interface():
    """Test interface verification."""
    interface = FunctionInterface(
        function_name="add_numbers",
        parameters=[
            Parameter(name="a", type="int", description="First number"),
            Parameter(name="b", type="int", description="Second number"),
        ],
        return_type="int",
        description="Add two numbers",
    )

    # Test function name mismatch
    generated_code = {
        "code": "def wrong_name(a: int, b: int) -> int: return a + b",
        "function_info": {
            "name": "wrong_name",
            "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "int",
        },
    }

    errors = verify_interface(interface, generated_code)
    assert "Function name mismatch" in errors[0]

    # Test valid interface
    generated_code["code"] = "def add_numbers(a: int, b: int) -> int: return a + b"
    generated_code["function_info"]["name"] = "add_numbers"

    errors = verify_interface(interface, generated_code)
    assert not errors
