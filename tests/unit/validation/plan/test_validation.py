"""Tests for plan validation functionality."""
import pytest

from evolia.models import (
    FunctionInterface,
    OutputDefinition,
    Parameter,
    Plan,
    PlanStep,
    SystemTool,
)
from evolia.utils.exceptions import PlanValidationError
from evolia.validation import validate_plan


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        "openai": {
            "model": "gpt-4o-2024-08-06",
            "api_key": "test-key",
            "temperature": 0.7,
        }
    }


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
        interface=FunctionInterface(
            function_name="test_function",
            parameters=[
                Parameter(name="input1", type="str", description="First input"),
                Parameter(name="input2", type="int", description="Second input"),
            ],
            return_type="str",
            description="Test tool function",
        ),
    )
    return {"test_tool": test_tool}


def test_validate_plan_basic(system_tools, mock_config):
    """Test basic plan validation."""
    plan = Plan(
        steps=[
            PlanStep(
                name="test_step",
                tool="test_tool",
                inputs={"input1": "test", "input2": 42},
                outputs={"result": OutputDefinition(type="str")},
                allowed_read_paths=[],
                allowed_write_paths=[],
                allowed_create_paths=[],
                default_policy="deny",
            )
        ],
        artifacts_dir="test_artifacts",
    )

    errors = validate_plan(plan, system_tools, mock_config)
    assert not errors


def test_validate_plan_unknown_tool(system_tools, mock_config):
    """Test validation with unknown tool."""
    plan = Plan(
        steps=[
            PlanStep(
                name="test_step",
                tool="unknown_tool",
                inputs={},
                outputs={},
                allowed_read_paths=[],
                allowed_write_paths=[],
                allowed_create_paths=[],
                default_policy="deny",
            )
        ],
        artifacts_dir="test_artifacts",
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_plan(plan, system_tools, mock_config)
    assert "Unknown tool" in str(exc_info.value)


def test_validate_plan_invalid_paths(system_tools, mock_config):
    """Test validation of file access paths."""
    plan = Plan(
        steps=[
            PlanStep(
                name="test_step",
                tool="test_tool",
                inputs={"input1": "test", "input2": 42},
                outputs={"result": OutputDefinition(type="str")},
                allowed_read_paths="invalid",  # Should be list
                allowed_write_paths=[],
                allowed_create_paths=[],
                default_policy="deny",
            )
        ],
        artifacts_dir="test_artifacts",
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_plan(plan, system_tools, mock_config)
    assert "allowed_read_paths must be a list" in str(exc_info.value)


def test_validate_plan_invalid_policy(system_tools, mock_config):
    """Test validation of default policy."""
    plan = Plan(
        steps=[
            PlanStep(
                name="test_step",
                tool="test_tool",
                inputs={"input1": "test", "input2": 42},
                outputs={"result": OutputDefinition(type="str")},
                allowed_read_paths=[],
                allowed_write_paths=[],
                allowed_create_paths=[],
                default_policy="invalid",  # Should be 'allow' or 'deny'
            )
        ],
        artifacts_dir="test_artifacts",
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_plan(plan, system_tools, mock_config)
    assert "default_policy must be 'allow' or 'deny'" in str(exc_info.value)
