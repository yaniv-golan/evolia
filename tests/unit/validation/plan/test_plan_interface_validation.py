"""Tests for plan interface validation."""
import pytest

from evolia.models import (
    FunctionInterface,
    OutputDefinition,
    Parameter,
    PlanStep,
    SystemTool,
)
from evolia.validation import validate_step_interface


@pytest.fixture
def system_tools():
    """Create test system tools."""
    test_tool = SystemTool(
        name="test_tool",
        description="Test tool",
        parameters=[
            Parameter(name="input1", type="str", description="First input"),
            Parameter(name="input2", type="int", description="Second input"),
        ],
        outputs={"result": OutputDefinition(type="str")},
        permissions={"read": [], "write": [], "create": []},
    )
    return {"test_tool": test_tool}


def test_system_tool_validation(system_tools):
    """Test validation of system tool interfaces."""
    # Test valid step
    step = PlanStep(
        name="test_step",
        tool="test_tool",
        inputs={"input1": "test", "input2": 42},
        outputs={"result": OutputDefinition(type="str")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny",
    )

    validation = validate_step_interface(step, system_tools)
    assert validation.matches_interface
    assert not validation.validation_errors


def test_system_tool_validation_missing_input(system_tools):
    """Test validation with missing input parameter."""
    step = PlanStep(
        name="test_step",
        tool="test_tool",
        inputs={"input1": "test"},  # Missing input2
        outputs={"result": OutputDefinition(type="str")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny",
    )

    validation = validate_step_interface(step, system_tools)
    assert not validation.matches_interface
    assert any(
        "Missing required inputs" in error for error in validation.validation_errors
    )


def test_system_tool_validation_wrong_output_type(system_tools):
    """Test validation with wrong output type."""
    step = PlanStep(
        name="test_step",
        tool="test_tool",
        inputs={"input1": "test", "input2": 42},
        outputs={"result": OutputDefinition(type="int")},  # Should be str
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny",
    )

    validation = validate_step_interface(step, system_tools)
    assert not validation.matches_interface
    assert any(
        "Output type mismatch" in error for error in validation.validation_errors
    )


def test_generate_code_validation():
    """Test validation of generate_code step."""
    step = PlanStep(
        name="generate_step",
        tool="generate_code",
        inputs={
            "function_name": "test_function",
            "parameters": [{"name": "x", "type": "int", "description": "Input value"}],
            "return_type": "int",
            "description": "Test function",
        },
        outputs={"code_file": OutputDefinition(type="str")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny",
    )

    validation = validate_step_interface(step, {})
    assert validation.matches_interface
    assert not validation.validation_errors


def test_execute_code_validation():
    """Test validation of execute_code step."""
    step = PlanStep(
        name="execute_step",
        tool="execute_code",
        inputs={"script_file": "test_artifacts/test_function.py", "x": 42},
        outputs={"result": OutputDefinition(type="int")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny",
    )

    validation = validate_step_interface(step, {})
    assert validation.matches_interface
    assert not validation.validation_errors


def test_execute_code_validation_invalid_script():
    """Test validation with invalid script file."""
    step = PlanStep(
        name="execute_step",
        tool="execute_code",
        inputs={"script_file": "test_function.txt"},  # Not a Python file
        outputs={"result": OutputDefinition(type="int")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny",
    )

    validation = validate_step_interface(step, {})
    assert not validation.matches_interface
    assert any(
        "must be a Python file" in error for error in validation.validation_errors
    )
