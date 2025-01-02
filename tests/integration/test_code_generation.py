"""Integration tests for code generation system."""

import pytest
from typing import Dict, Any
from unittest.mock import patch

from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.core.code_fixer import CodeFixer, CodeFixConfig
from evolia.core.function_generator import FunctionGenerator

@pytest.fixture
def mock_openai():
    """Mock OpenAI structured output calls."""
    with patch('evolia.core.code_generator.call_openai_structured') as mock:
        def side_effect(api_key, model, json_schema, user_prompt, system_prompt=None, **kwargs):
            if 'error_msg' in user_prompt and 'ZeroDivisionError' in user_prompt:
                # Code fixing case
                return {
                    'code': 'def divide(a: int, b: int) -> float:\n    if b == 0:\n        raise ValueError("Cannot divide by zero")\n    return a / b',
                    'validation_results': {
                        'syntax_valid': True,
                        'security_issues': []
                    },
                    'fix_description': 'Added check for division by zero',
                    'outputs': {
                        'result': {
                            'type': 'float',
                            'reference': '$divide.result'
                        }
                    }
                }
            else:
                # Function generation case
                return {
                    'code': 'def add_nums(a: int, b: int) -> int:\n    return a + b',
                    'validation_results': {
                        'syntax_valid': True,
                        'security_issues': []
                    },
                    'function_info': {
                        'name': 'add_nums',
                        'parameters': [
                            {'name': 'a', 'type': 'int'},
                            {'name': 'b', 'type': 'int'}
                        ],
                        'return_type': 'int',
                        'docstring': 'Add two numbers.'
                    },
                    'outputs': {
                        'sum': {
                            'type': 'int',
                            'reference': '$add_nums.sum'
                        }
                    }
                }
        mock.side_effect = side_effect
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

def test_generate_function(mock_openai, function_generator):
    """Test generating a function with constraints."""
    response = function_generator.generate_function(
        requirements="Add two numbers and validate inputs",
        function_name="add_nums",
        parameters=[
            {'name': 'a', 'type': 'int'},
            {'name': 'b', 'type': 'int'}
        ],
        return_type="int"
    )
    
    assert response['validation_results']['syntax_valid']
    assert 'def add_nums' in response['code']
    assert 'int' in response['code']
    
    # Execute the code
    namespace = {}
    exec(response['code'], namespace)
    assert namespace['add_nums'](2, 3) == 5

def test_code_fixing(mock_openai, code_fixer):
    """Test fixing code with error."""
    original_code = """
def divide(a: int, b: int) -> float:
    return a / b
"""
    
    response = code_fixer.fix_code(
        code=original_code,
        error_msg="ZeroDivisionError: division by zero"
    )
    
    assert response['validation_results']['syntax_valid']
    assert 'def divide' in response['code']
    assert 'if' in response['code'].lower()  # Should have error checking

def test_error_handling(mock_openai, function_generator, code_fixer):
    """Test error handling in generation and fixing."""
    # Test generation error
    mock_openai.side_effect = Exception("API Error")
    
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