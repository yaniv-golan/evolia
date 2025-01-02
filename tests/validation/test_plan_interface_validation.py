"""Tests for plan interface validation."""
import pytest
import argparse
from pathlib import Path
from typing import Any, Dict, List

from evolia.core.evolia import validate_step_interface
from evolia.models import PlanStep, SystemTool, Parameter, OutputDefinition

def types_are_compatible(type1: str, type2: str) -> bool:
    """Check if two types are compatible.
    
    Args:
        type1: First type
        type2: Second type
        
    Returns:
        Whether types are compatible
    """
    # For now, just check exact match
    return type1 == type2

def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier.
    
    Args:
        name: String to check
        
    Returns:
        Whether string is a valid identifier
    """
    return name.isidentifier()

@pytest.fixture
def mock_args():
    """Mock command line arguments."""
    args = argparse.Namespace()
    args.allow_read = []
    args.allow_write = []
    args.allow_create = []
    args.default_policy = "deny"
    args.keep_artifacts = True
    args.ephemeral_dir = str(Path.cwd() / "artifacts")
    return args

def test_system_tool_validation():
    """Test validation of system tool steps."""
    # Create test tool
    tool = SystemTool(
        name="test_tool",
        description="A test tool",
        parameters=[
            Parameter(name="x", type="int", description="First number"),
            Parameter(name="y", type="int", description="Second number")
        ],
        outputs={"result": OutputDefinition(type="int")}
    )
    
    # Create test step
    step = PlanStep(
        name="Test step",
        tool="test_tool",
        inputs={"x": 5, "y": 3},
        outputs={"result": OutputDefinition(type="int")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny"
    )
    
    # Validate step
    validation = validate_step_interface(step, {"test_tool": tool})
    assert validation.matches_interface
    assert not validation.validation_errors

def test_generate_code_validation():
    """Test validation of generate_code steps."""
    step = PlanStep(
        name="Generate function",
        tool="generate_code",
        inputs={
            "function_name": "test_function",
            "parameters": [
                {"name": "x", "type": "int", "description": "First number"}
            ],
            "return_type": "int",
            "description": "A test function"
        },
        outputs={"code_file": OutputDefinition(type="str")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny"
    )
    
    validation = validate_step_interface(step, {})
    assert validation.matches_interface
    assert not validation.validation_errors

def test_execute_code_validation():
    """Test validation of execute_code steps."""
    step = PlanStep(
        name="Execute function",
        tool="execute_code",
        inputs={
            "script_file": "test.py",
            "x": 5
        },
        outputs={"result": OutputDefinition(type="int")},
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_create_paths=[],
        default_policy="deny"
    )
    
    validation = validate_step_interface(step, {})
    assert validation.matches_interface
    assert not validation.validation_errors 