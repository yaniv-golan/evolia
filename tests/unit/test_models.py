import pytest
from pydantic import ValidationError
from evolia.models import (
    Plan,
    PlanStep,
    Parameter,
    FunctionInterface,
    CodeGenerationRequest,
    ValidationResults
)

def test_parameter_validation():
    """Test Parameter model validation"""
    # Valid parameter
    param = Parameter(
        name="input_str",
        type="str",
        description="Input string to process"
    )
    assert param.name == "input_str"
    assert param.type == "str"
    assert param.description == "Input string to process"
    
    # Invalid parameter name
    with pytest.raises(ValidationError) as exc:
        Parameter(name="2invalid", type="str")
    assert "Invalid parameter name" in str(exc.value)
    
    # Invalid type annotation
    with pytest.raises(ValidationError) as exc:
        Parameter(name="valid", type="invalid:type::")
    assert "Invalid type annotation" in str(exc.value)

def test_code_generation_request():
    """Test CodeGenerationRequest validation"""
    # Test valid request
    request = CodeGenerationRequest(
        function_name="process_data",
        parameters=[
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to input file"
            }
        ],
        return_type="str",
        description="Process data from a file",
        examples=["process_data('input.csv')"],
        constraints=["no_globals", "pure_function"]
    )
    
    assert request.function_name == "process_data"
    assert len(request.parameters) == 1
    assert request.return_type == "str"
    assert request.description == "Process data from a file"
    assert request.examples == ["process_data('input.csv')"]
    assert request.constraints == ["no_globals", "pure_function"]
    
    # Test missing required fields
    with pytest.raises(ValidationError):
        CodeGenerationRequest(
            parameters=[{"name": "input_path", "type": "str"}],
            return_type="str"
        )
    
    # Invalid function name
    with pytest.raises(ValidationError) as exc:
        CodeGenerationRequest(
            function_name="1invalid",
            parameters=[Parameter(name="x", type="int")],
            return_type="int"
        )
    assert "Invalid function name" in str(exc.value)
    
    # Invalid constraint
    with pytest.raises(ValidationError) as exc:
        CodeGenerationRequest(
            function_name="valid",
            parameters=[Parameter(name="x", type="int")],
            return_type="int",
            constraints=["invalid_constraint"]
        )
    assert "Invalid constraints" in str(exc.value)

def test_test_results():
    """Test TestResults model"""
    failure = TestFailure(
        test_case={"input": 5, "expected": 10},
        expected=10,
        actual=5,
        error="AssertionError: expected 10 but got 5"
    )
    
    results = TestResults(
        passed=False,
        failures=[failure],
        execution_time=0.123
    )
    assert not results.passed
    assert len(results.failures) == 1
    assert results.execution_time == 0.123
    assert results.failures[0].error == "AssertionError: expected 10 but got 5"

def test_validation_results():
    """Test ValidationResults model"""
    results = ValidationResults(
        syntax_valid=True,
        name_matches=True,
        params_match=False,
        return_type_matches=True,
        security_issues=["Uses unsafe eval()"]
    )
    assert results.syntax_valid
    assert results.name_matches
    assert not results.params_match
    assert results.return_type_matches
    assert len(results.security_issues) == 1

def test_generated_code():
    """Test GeneratedCode model"""
    validation = ValidationResults(
        syntax_valid=True,
        name_matches=True,
        params_match=True,
        return_type_matches=True,
        security_issues=[]
    )
    
    test_results = TestResults(
        passed=True,
        failures=[],
        execution_time=0.05
    )
    
    code = GeneratedCode(
        code="def example(x: int) -> int:\n    return x * 2",
        function_name="example",
        validation_results=validation,
        test_results=test_results
    )
    
    assert code.code.startswith("def example")
    assert code.function_name == "example"
    assert code.validation_results.syntax_valid
    assert code.test_results.passed
    
    # Test model_dump conversion and logging
    data = code.model_dump()
    assert "code" in data
    assert "validation_results" in data
    assert "test_results" in data 