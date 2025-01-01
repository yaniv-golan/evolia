"""Unit tests for FunctionGenerator."""

import pytest
from unittest.mock import Mock, patch

from evolia.core.function_generator import FunctionGenerator
from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig

@pytest.fixture
def mock_code_generator():
    """Create a mock CodeGenerator."""
    generator = Mock(spec=CodeGenerator)
    # Create a real config object for the mock
    generator.config = CodeGenerationConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=100,
        allowed_modules={'math', 'typing'},
        allowed_builtins={'len', 'str'},
        api_key="test-key"
    )
    generator.generate.return_value = {
        'code': 'def test(x: int) -> int:\n    return x + 1',
        'validation_results': {
            'syntax_valid': True,
            'security_issues': []
        },
        'function_info': {
            'name': 'test',
            'parameters': [{'name': 'x', 'type': 'int'}],
            'return_type': 'int',
            'docstring': 'Test function.'
        }
    }
    return generator

@pytest.fixture
def function_generator(mock_code_generator):
    """Create a FunctionGenerator instance."""
    return FunctionGenerator(mock_code_generator)

def test_function_generator_init(mock_code_generator):
    """Test FunctionGenerator initialization."""
    generator = FunctionGenerator(mock_code_generator)
    assert generator.code_generator == mock_code_generator
    assert generator.function_template is not None
    assert generator.function_schema is not None

def test_generate_function_success(function_generator):
    """Test successful function generation."""
    response = function_generator.generate_function(
        requirements="Add one to the input number",
        function_name="add_one",
        parameters=[{'name': 'x', 'type': 'int'}],
        return_type="int"
    )
    
    assert response['code'] == 'def test(x: int) -> int:\n    return x + 1'
    assert response['validation_results']['syntax_valid']
    assert response['function_info']['name'] == 'test'
    assert len(response['function_info']['parameters']) == 1

def test_generate_function_with_context(function_generator):
    """Test function generation with additional context."""
    function_generator.generate_function(
        requirements="Add numbers safely",
        function_name="add_safe",
        parameters=[
            {'name': 'a', 'type': 'int'},
            {'name': 'b', 'type': 'int'}
        ],
        return_type="int",
        context="Handle integer overflow"
    )
    
    # Verify context was included in prompt
    call_args = function_generator.code_generator.generate.call_args[1]
    template_vars = call_args['template_vars']
    assert "Handle integer overflow" in template_vars['context']

def test_generate_function_parameter_formatting(function_generator):
    """Test parameter formatting in function generation."""
    function_generator.generate_function(
        requirements="Test function",
        function_name="test",
        parameters=[
            {'name': 'a', 'type': 'int'},
            {'name': 'b', 'type': 'str'},
            {'name': 'c', 'type': 'bool'}
        ],
        return_type="Dict[str, Any]"
    )
    
    # Verify parameters were formatted correctly
    call_args = function_generator.code_generator.generate.call_args[1]
    template_vars = call_args['template_vars']
    params_str = template_vars['parameters']
    assert "a: int" in params_str
    assert "b: str" in params_str
    assert "c: bool" in params_str

def test_generate_function_error_handling(function_generator):
    """Test error handling in function generation."""
    function_generator.code_generator.generate.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        function_generator.generate_function(
            requirements="Test function",
            function_name="test",
            parameters=[{'name': 'x', 'type': 'int'}],
            return_type="int"
        )
    
    assert str(exc_info.value) == "Test error"

def test_generate_function_template_variables(function_generator):
    """Test all template variables are properly set."""
    requirements = "Test requirements"
    function_name = "test_func"
    parameters = [{'name': 'x', 'type': 'int'}]
    return_type = "str"
    context = "Test context"
    
    function_generator.generate_function(
        requirements=requirements,
        function_name=function_name,
        parameters=parameters,
        return_type=return_type,
        context=context
    )
    
    call_args = function_generator.code_generator.generate.call_args[1]
    template_vars = call_args['template_vars']
    
    assert template_vars['requirements'] == requirements
    assert template_vars['function_name'] == function_name
    assert template_vars['return_type'] == return_type
    assert template_vars['context'] == context
    assert template_vars['allowed_modules'] == sorted(function_generator.code_generator.config.allowed_modules)
    assert template_vars['allowed_builtins'] == sorted(function_generator.code_generator.config.allowed_builtins)
    assert 'cot' in template_vars 