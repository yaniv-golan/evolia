"""Unit tests for tool metadata."""

from typing import Any, Dict

import pytest

from evolia.models.models import FunctionInterface, Parameter, SystemTool


@pytest.fixture
def test_tool_data():
    """Create test tool data."""
    return {
        "name": "test_tool",
        "description": "A test tool for unit testing",
        "parameters": [
            {"name": "x", "type": "int", "description": "First number"},
            {"name": "y", "type": "int", "description": "Second number"},
        ],
        "outputs": {"result": {"type": "int", "description": "The sum of x and y"}},
    }


def test_system_tool_creation(test_tool_data):
    """Test system tool creation."""
    tool = SystemTool(**test_tool_data)

    # Test required fields
    assert tool.name == "test_tool"
    assert tool.description == "A test tool for unit testing"

    # Test parameters
    params = tool.parameters
    assert len(params) == 2
    assert params[0]["name"] == "x"
    assert params[0]["type"] == "int"
    assert params[1]["name"] == "y"
    assert params[1]["type"] == "int"

    # Test outputs
    assert len(tool.outputs) == 1
    assert tool.outputs["result"]["type"] == "int"


def test_parameter_validation():
    """Test parameter validation."""
    # Test valid parameter
    param = Parameter(name="test", type="int", description="A test parameter")
    assert param.name == "test"
    assert param.type == "int"

    # Test invalid type
    with pytest.raises(ValueError, match="Invalid parameter type"):
        Parameter(name="test", type="invalid_type", description="Invalid type")
