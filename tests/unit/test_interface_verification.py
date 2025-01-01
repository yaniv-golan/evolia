"""Tests for interface verification functionality."""

import pytest
from dataclasses import dataclass
from typing import Dict, Any, List

from evolia.core.interface_verification import verify_interface

@dataclass
class MockInterface:
    """Mock interface for testing."""
    function_name: str
    parameters: List[Any]
    return_type: str
    outputs: Dict[str, Any]
    docstring_required: bool = False
    examples: List[Dict[str, Any]] = None
    constraints: List[str] = None

@dataclass
class MockParameter:
    """Mock parameter for testing."""
    name: str
    type: str
    description: str = ""

@dataclass
class MockOutput:
    """Mock output for testing."""
    type: str
    reference: str = None

def test_verify_interface_valid_outputs():
    """Test interface verification with valid outputs."""
    interface = MockInterface(
        function_name="add_numbers",
        parameters=[
            MockParameter(name="a", type="int"),
            MockParameter(name="b", type="int")
        ],
        return_type="str",
        outputs={
            "result": MockOutput(type="str", reference="$add_numbers.result")
        }
    )
    
    response = {
        "function_name": "add_numbers",
        "parameters": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ],
        "return_type": "str",
        "outputs": {
            "result": {
                "type": "str",
                "reference": "$add_numbers.result"
            }
        }
    }
    
    errors = verify_interface(response, interface)
    assert not errors, f"Expected no errors but got: {errors}"

def test_verify_interface_missing_output():
    """Test interface verification with missing output."""
    interface = MockInterface(
        function_name="add_numbers",
        parameters=[
            MockParameter(name="a", type="int"),
            MockParameter(name="b", type="int")
        ],
        return_type="str",
        outputs={
            "result": MockOutput(type="str", reference="$add_numbers.result")
        }
    )
    
    response = {
        "function_name": "add_numbers",
        "parameters": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ],
        "return_type": "str",
        "outputs": {}
    }
    
    errors = verify_interface(response, interface)
    assert errors
    assert any("Missing output: result" in error for error in errors)

def test_verify_interface_wrong_output_type():
    """Test interface verification with wrong output type."""
    interface = MockInterface(
        function_name="add_numbers",
        parameters=[
            MockParameter(name="a", type="int"),
            MockParameter(name="b", type="int")
        ],
        return_type="str",
        outputs={
            "result": MockOutput(type="str", reference="$add_numbers.result")
        }
    )
    
    response = {
        "function_name": "add_numbers",
        "parameters": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ],
        "return_type": "str",
        "outputs": {
            "result": {
                "type": "int",  # Wrong type
                "reference": "$add_numbers.result"
            }
        }
    }
    
    errors = verify_interface(response, interface)
    assert errors
    assert any("Output type mismatch for result" in error for error in errors)

def test_verify_interface_invalid_reference_format():
    """Test interface verification with invalid reference format."""
    interface = MockInterface(
        function_name="add_numbers",
        parameters=[
            MockParameter(name="a", type="int"),
            MockParameter(name="b", type="int")
        ],
        return_type="str",
        outputs={
            "result": MockOutput(type="str", reference="$add_numbers.result")
        }
    )
    
    response = {
        "function_name": "add_numbers",
        "parameters": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ],
        "return_type": "str",
        "outputs": {
            "result": {
                "type": "str",
                "reference": "invalid_reference"  # Invalid format
            }
        }
    }
    
    errors = verify_interface(response, interface)
    assert errors
    assert any("Invalid output reference format" in error for error in errors)

def test_verify_interface_multiple_outputs():
    """Test interface verification with multiple outputs."""
    interface = MockInterface(
        function_name="process_data",
        parameters=[
            MockParameter(name="data", type="dict")
        ],
        return_type="dict",
        outputs={
            "processed": MockOutput(type="dict", reference="$process_data.processed"),
            "metadata": MockOutput(type="dict", reference="$process_data.metadata")
        }
    )
    
    response = {
        "function_name": "process_data",
        "parameters": [
            {"name": "data", "type": "dict"}
        ],
        "return_type": "dict",
        "outputs": {
            "processed": {
                "type": "dict",
                "reference": "$process_data.processed"
            },
            "metadata": {
                "type": "dict",
                "reference": "$process_data.metadata"
            }
        }
    }
    
    errors = verify_interface(response, interface)
    assert not errors, f"Expected no errors but got: {errors}" 