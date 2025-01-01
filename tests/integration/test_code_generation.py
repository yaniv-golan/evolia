"""Integration tests for code generation system."""

import pytest
from typing import Dict, Any
from unittest.mock import patch

from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.core.code_fixer import CodeFixer, CodeFixConfig
from evolia.core.function_generator import FunctionGenerator
from evolia.core.code_generator import call_openai_structured

@pytest.fixture
def mock_openai():
    """Mock OpenAI structured output calls."""
    with patch('evolia.core.code_generator.call_openai_structured') as mock:
        mock.return_value = {
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
            },
            'fix_description': 'Fixed the issue'
        }
        yield mock

@pytest.fixture
def code_generator(mock_openai):
    """Create a CodeGenerator with test configuration."""
    config = CodeGenerationConfig(
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        max_tokens=500,
        allowed_modules={'math', 'typing'},
        allowed_builtins={'len', 'str', 'int', 'float'},
        api_key="test-key"
    )
    return CodeGenerator(config)

@pytest.fixture
def function_generator(code_generator):
    """Create a FunctionGenerator instance."""
    return FunctionGenerator(code_generator)

@pytest.fixture
def code_fixer(code_generator):
    """Create a CodeFixer instance."""
    config = CodeFixConfig(fix_temperature=0.1, max_attempts=2)
    return CodeFixer(code_generator, config)

def test_generate_and_fix_function(mock_openai, function_generator, code_fixer):
    """Test generating a function and then fixing it."""
    # Generate a function
    response = function_generator.generate_function(
        requirements="Add two numbers, but throw an error if the sum would overflow",
        function_name="add_safe",
        parameters=[
            {'name': 'a', 'type': 'int'},
            {'name': 'b', 'type': 'int'}
        ],
        return_type="int"
    )
    
    generated_code = response['code']
    assert 'def test' in generated_code
    assert 'int' in generated_code
    assert response['validation_results']['syntax_valid']
    
    # Simulate an error in the generated code
    error_msg = "OverflowError: integer addition result too large for a 64-bit integer"
    
    # Fix the code
    fix_response = code_fixer.fix_code(
        code=generated_code,
        error_msg=error_msg
    )
    
    fixed_code = fix_response['code']
    assert 'def test' in fixed_code
    assert response['validation_results']['syntax_valid']

def test_function_generation_with_constraints(mock_openai):
    """Test generating a function with specific constraints."""
    code_generator = CodeGenerator(CodeGenerationConfig(
        temperature=0.2,
        allowed_modules={'math'},
        allowed_builtins={'int', 'float'},
        api_key="test-key"
    ))
    function_generator = FunctionGenerator(code_generator)
    
    response = function_generator.generate_function(
        requirements="Calculate the area of a circle",
        function_name="circle_area",
        parameters=[{'name': 'radius', 'type': 'float'}],
        return_type="float",
        context="Use math.pi for precise calculations"
    )
    
    generated_code = response['code']
    assert 'def test' in generated_code
    assert response['validation_results']['syntax_valid']

def test_code_fixing_with_history(mock_openai):
    """Test fixing code with multiple attempts."""
    code_generator = CodeGenerator(CodeGenerationConfig(api_key="test-key"))
    code_fixer = CodeFixer(code_generator, CodeFixConfig(max_attempts=3))
    
    original_code = """
def divide(a: int, b: int) -> float:
    return a / b
"""
    
    # First fix attempt
    response1 = code_fixer.fix_code(
        code=original_code,
        error_msg="ZeroDivisionError: division by zero"
    )
    
    fixed_code1 = response1['code']
    assert 'def test' in fixed_code1
    assert response1['validation_results']['syntax_valid']
    
    # Second fix attempt with a different error
    response2 = code_fixer.fix_code(
        code=fixed_code1,
        error_msg="TypeError: 'float' object cannot be interpreted as an integer"
    )
    
    fixed_code2 = response2['code']
    assert 'def test' in fixed_code2
    assert response2['validation_results']['syntax_valid']
    
    # Verify fix history
    assert len(code_fixer.fix_history.attempts) == 2

def test_generate_function_with_docstring(mock_openai):
    """Test generating a function with comprehensive docstring."""
    code_generator = CodeGenerator(CodeGenerationConfig(api_key="test-key"))
    function_generator = FunctionGenerator(code_generator)
    
    response = function_generator.generate_function(
        requirements="Convert a temperature from Celsius to Fahrenheit",
        function_name="celsius_to_fahrenheit",
        parameters=[{'name': 'celsius', 'type': 'float'}],
        return_type="float"
    )
    
    generated_code = response['code']
    assert 'def test' in generated_code
    assert response['validation_results']['syntax_valid']
    assert response['function_info']['docstring']

def test_error_propagation():
    """Test error handling and propagation between components."""
    with patch('evolia.core.code_generator.call_openai_structured') as mock:
        mock.side_effect = Exception("Test error")
        
        code_generator = CodeGenerator(CodeGenerationConfig(
            model="test-model",
            temperature=0.2,
            api_key="test-key"
        ))
        function_generator = FunctionGenerator(code_generator)
        code_fixer = CodeFixer(code_generator)
        
        # Test function generation error
        with pytest.raises(Exception):
            function_generator.generate_function(
                requirements="Test function",
                function_name="test",
                parameters=[{'name': 'x', 'type': 'int'}],
                return_type="int"
            )
        
        # Test fix error
        with pytest.raises(ValueError):
            code_fixer.fix_code(
                code="def test(): pass",
                error_msg="Test error"
            ) 