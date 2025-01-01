"""Tests for plan interface validation functionality"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from evolia.core.evolia import generate_plan
from evolia.models import Plan, PlanStep
from evolia.utils.exceptions import PlanValidationError

# Mock system tools for testing
MOCK_SYSTEM_TOOLS = [
    {
        "name": "test_function",
        "version": "1.0.0",
        "description": "A test tool",
        "interface": {
            "function_name": "test_function",
            "parameters": [
                {"name": "input_text", "type": "str", "description": "Input text to process"},
                {"name": "max_length", "type": "int", "description": "Maximum length"}
            ],
            "return_type": "str",
            "description": "Test function description",
            "examples": ["test_function('hello', 5)"],
            "constraints": ["no_globals", "pure_function"]
        }
    }
]

# Mock OpenAI response
MOCK_OPENAI_RESPONSE = {
    "steps": [
        {
            "name": "Process Text",
            "tool": "test_function",
            "inputs": {
                "input_text": "test input",
                "max_length": 10
            },
            "outputs": {
                "result": "str"
            },
            "allowed_read_paths": [],
            "interface_validation": {
                "matches_interface": True,
                "validation_errors": []
            }
        }
    ]
}

@pytest.fixture
def mock_openai():
    with patch('evolia.core.evolia.call_openai_structured') as mock:
        mock.return_value = {
            "steps": [
                {
                    "name": "Process Text",
                    "tool": "test_function",
                    "inputs": {
                        "input_text": "test input",
                        "max_length": 10
                    },
                    "outputs": {
                        "result": "str"
                    },
                    "allowed_read_paths": [],
                    "interface_validation": {
                        "matches_interface": True,
                        "validation_errors": []
                    }
                }
            ]
        }
        yield mock

def test_generate_plan_validates_interfaces(mock_openai):
    """Test that generate_plan validates tool interfaces"""
    config = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model": "gpt-4",
            "max_retries": 3
        }
    }
    
    # Mock args
    args = argparse.Namespace(
        allow_read=[],
        allow_write=[],
        allow_create=[],
        default_policy="deny"
    )
    
    plan = generate_plan("Test task", MOCK_SYSTEM_TOOLS, config, args)
    assert plan is not None
    assert len(plan.steps) > 0
    
    # Verify each step has interface validation
    for step in plan.steps:
        assert step.interface_validation is not None
        assert step.interface_validation.matches_interface
        assert not step.interface_validation.validation_errors

def test_generate_plan_catches_interface_mismatch(mock_openai):
    """Test that generate_plan catches interface mismatches"""
    config = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model": "gpt-4",
            "max_retries": 3
        }
    }
    
    # Mock args
    args = argparse.Namespace(
        allow_read=[],
        allow_write=[],
        allow_create=[],
        default_policy="deny"
    )
    
    plan = generate_plan("Test task", MOCK_SYSTEM_TOOLS, config, args)
    assert plan is not None
    assert len(plan.steps) > 0
    
    # Verify interface validation caught the mismatch
    found_mismatch = False
    for step in plan.steps:
        if not step.interface_validation.matches_interface:
            found_mismatch = True
            assert len(step.interface_validation.validation_errors) > 0
    
    assert found_mismatch, "Expected at least one interface mismatch"

def test_generate_plan_validates_generate_code_interface(mock_openai):
    """Test that generate_plan validates generate_code interface"""
    config = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model": "gpt-4",
            "max_retries": 3
        }
    }
    
    # Mock args
    args = argparse.Namespace(
        allow_read=[],
        allow_write=[],
        allow_create=[],
        default_policy="deny"
    )
    
    plan = generate_plan("Test task", MOCK_SYSTEM_TOOLS, config, args)
    assert plan is not None
    
    # Find generate_code step
    generate_step = next(
        (step for step in plan.steps if step.tool == "generate_code"),
        None
    )
    assert generate_step is not None
    
    # Verify interface validation
    assert generate_step.interface_validation is not None
    assert generate_step.interface_validation.matches_interface
    assert not generate_step.interface_validation.validation_errors

def test_generate_plan_validates_execute_code_interface(mock_openai):
    """Test that generate_plan validates execute_code interface"""
    config = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model": "gpt-4",
            "max_retries": 3
        }
    }
    
    # Mock args
    args = argparse.Namespace(
        allow_read=[],
        allow_write=[],
        allow_create=[],
        default_policy="deny"
    )
    
    plan = generate_plan("Test task", MOCK_SYSTEM_TOOLS, config, args)
    assert plan is not None
    
    # Find execute_code step
    execute_step = next(
        (step for step in plan.steps if step.tool == "execute_code"),
        None
    )
    assert execute_step is not None
    
    # Verify interface validation
    assert execute_step.interface_validation is not None
    assert execute_step.interface_validation.matches_interface
    assert not execute_step.interface_validation.validation_errors

def test_system_tool_validation():
    """Test validation of system tool interfaces."""
    # Mock system tools
    system_tools = [{
        "name": "test_function",
        "interface": {
            "function_name": "test_function",
            "parameters": [
                {"name": "param1", "type": "str", "description": "First parameter"},
                {"name": "param2", "type": "int", "description": "Second parameter", "optional": True}
            ],
            "return_type": "str",
            "description": "A test tool"
        }
    }]
    
    # Test valid step
    step = PlanStep(
        name="Test Step",
        tool="test_function",
        inputs={"param1": "test"},
        outputs={"result": "str"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, system_tools)
    assert validation.matches_interface
    assert not validation.validation_errors
    
    # Test missing required parameter
    step = PlanStep(
        name="Test Step",
        tool="test_function",
        inputs={},
        outputs={"result": "str"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, system_tools)
    assert not validation.matches_interface
    assert any("Missing parameter: 'param1'" in err for err in validation.validation_errors)
    
    # Test extra parameter
    step = PlanStep(
        name="Test Step",
        tool="test_function",
        inputs={"param1": "test", "extra": "value"},
        outputs={"result": "str"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, system_tools)
    assert not validation.matches_interface
    assert any("Extra parameters" in err for err in validation.validation_errors)
    
    # Test type mismatch
    step = PlanStep(
        name="Test Step",
        tool="test_function",
        inputs={"param1": 123},  # Should be str
        outputs={"result": "str"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, system_tools)
    assert not validation.matches_interface
    assert any("Type mismatch" in err for err in validation.validation_errors)

def test_generate_code_validation():
    """Test validation of generate_code interfaces."""
    # Test valid step
    step = PlanStep(
        name="Generate Function",
        tool="generate_code",
        inputs={
            "function_name": "process_data",
            "parameters": [{"name": "input_file", "type": "str"}],
            "return_type": "Dict[str, Any]",
            "description": "Process data from file"
        },
        outputs={"code_file": "run_artifacts/process_data.py"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, [])
    assert validation.matches_interface
    assert not validation.validation_errors
    
    # Test missing required field
    step = PlanStep(
        name="Generate Function",
        tool="generate_code",
        inputs={
            "function_name": "process_data",
            "parameters": [{"name": "input_file", "type": "str"}],
            "return_type": "Dict[str, Any]"
            # Missing description
        },
        outputs={"code_file": "run_artifacts/process_data.py"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, [])
    assert not validation.matches_interface
    assert any("Missing required fields" in err for err in validation.validation_errors)
    
    # Test invalid function name
    with pytest.raises(ValueError, match="Invalid function name: 1invalid"):
        step = PlanStep(
            name="Generate Function",
            tool="generate_code",
            inputs={
                "function_name": "1invalid",
                "parameters": [{"name": "input_file", "type": "str"}],
                "return_type": "Dict[str, Any]",
                "description": "Process data from file"
            },
            outputs={"code_file": "run_artifacts/process_data.py"},
            allowed_read_paths=[],
            interface_validation=None
        )

def test_execute_code_validation():
    """Test validation of execute_code interfaces."""
    # Test valid step
    step = PlanStep(
        name="Execute Function",
        tool="execute_code",
        inputs={"script_file": "run_artifacts/process_data.py"},
        outputs={"result": "Dict[str, Any]"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, [])
    assert validation.matches_interface
    assert not validation.validation_errors
    
    # Test missing script_file
    step = PlanStep(
        name="Execute Function",
        tool="execute_code",
        inputs={},
        outputs={"result": "Dict[str, Any]"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, [])
    assert not validation.matches_interface
    assert any("Missing required field: script_file" in err for err in validation.validation_errors)
    
    # Test invalid script path
    step = PlanStep(
        name="Execute Function",
        tool="execute_code",
        inputs={"script_file": "/invalid/path.py"},
        outputs={"result": "Dict[str, Any]"},
        allowed_read_paths=[],
        interface_validation=None
    )
    validation = validate_step_interface(step, [])
    assert not validation.matches_interface
    assert any("Invalid script_file path" in err for err in validation.validation_errors)

def test_types_are_compatible():
    """Test type compatibility checking."""
    assert types_are_compatible("str", "str")
    assert types_are_compatible("int", "int")
    assert types_are_compatible("Any", "str")
    assert types_are_compatible("None", "Optional[str]")
    assert types_are_compatible("List[str]", "List[str]")
    assert types_are_compatible("Dict[str, Any]", "Dict[str, Any]")
    assert not types_are_compatible("str", "int")
    assert not types_are_compatible("List[str]", "List[int]")
    assert not types_are_compatible("Dict[str, str]", "Dict[str, int]")

def test_is_valid_identifier():
    """Test Python identifier validation."""
    assert is_valid_identifier("valid_name")
    assert is_valid_identifier("ValidName")
    assert is_valid_identifier("_private")
    assert not is_valid_identifier("1invalid")
    assert not is_valid_identifier("invalid-name")
    assert not is_valid_identifier("invalid name")
    assert not is_valid_identifier("") 