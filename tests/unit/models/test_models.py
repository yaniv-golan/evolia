"""Unit tests for models."""

import pytest
from typing import Dict, Any, List

from evolia.models.models import (
    Parameter,
    FunctionInterface,
    CodeGenerationRequest,
    ValidationResults,
    GeneratedCode, 
    OutputDefinition,
    CodeGenerationResponse
)

def test_parameter_validation():
    """Test parameter validation."""
    # Valid parameters
    param = Parameter(name="valid_name", type="str", description="A valid parameter")
    assert param.name == "valid_name"
    assert param.type == "str"
    assert param.description == "A valid parameter"
    
    # Invalid parameter name
    with pytest.raises(ValueError, match="Invalid parameter name"):
        Parameter(name="2invalid", type="str", description="Invalid name")
    
    # Missing required fields
    with pytest.raises(TypeError):
        Parameter(name="test")

def test_function_interface():
    """Test function interface validation."""
    params = [
        Parameter(name="x", type="int", description="First number"),
        Parameter(name="y", type="int", description="Second number")
    ]
    
    interface = FunctionInterface(
        function_name="add_numbers",
        parameters=params,
        return_type="int",
        description="Add two numbers"
    )
    
    assert interface.function_name == "add_numbers"
    assert len(interface.parameters) == 2
    assert interface.return_type == "int"
    assert interface.description == "Add two numbers"

def test_code_generation_request():
    """Test code generation request model."""
    request = CodeGenerationRequest(
        function_name="test_func",
        description="A test function",
        parameters=[
            Parameter(name="x", type="int", description="Test parameter")
        ],
        return_type="str"
    )
    
    assert request.function_name == "test_func"
    assert len(request.parameters) == 1
    assert request.return_type == "str"
    assert request.description == "A test function"

def test_validation_results():
    """Test validation results model."""
    results = ValidationResults(
        syntax_valid=True,
        security_issues=[]
    )
    
    assert results.syntax_valid
    assert not results.security_issues

def test_generated_code():
    """Test generated code model."""
    validation = ValidationResults(
        syntax_valid=True,
        security_issues=[]
    )
        
    code = CodeGenerationResponse(
        code="def test(): pass",
        validation_results=validation,
        outputs={"result": OutputDefinition(type="str")},
        function_name="test",
        parameters=[],
        return_type="None",
        description="Test function"
    )
    
    assert code.code == "def test(): pass"
    assert code.validation_results.syntax_valid
    assert code.function_name == "test"