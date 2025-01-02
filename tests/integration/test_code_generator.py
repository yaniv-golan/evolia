"""Integration tests for the code generator module."""

import os
import pytest
from typing import List, Dict, Any
from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.core.function_generator import FunctionGenerator
from evolia.models.models import Parameter
from evolia.utils.exceptions import CodeGenerationError

@pytest.fixture
def openai_api_key():
    """Ensure OPENAI_API_KEY is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key

@pytest.fixture
def function_generator(openai_api_key):
    """Create a function generator instance."""
    config = CodeGenerationConfig(
        api_key=openai_api_key,
        model="gpt-4o-2024-08-06",
        temperature=0.2,  # Low temperature for consistent results
        max_tokens=1000,
        allowed_modules={'math', 'typing', 'datetime', 're', 'json'},
        allowed_builtins={'len', 'str', 'int', 'float', 'list', 'dict', 'set'}
    )
    code_generator = CodeGenerator(config)
    return FunctionGenerator(code_generator)

@pytest.mark.integration
def test_math_function_generation(function_generator):
    """Test generating a simple math function."""
    response = function_generator.generate_function(
        requirements="Add two numbers and return their sum. Throw an error if either number is negative.",
        function_name="add_positive_numbers",
        parameters=[
            Parameter(name="a", type="float", description="First number (must be positive)"),
            Parameter(name="b", type="float", description="Second number (must be positive)")
        ],
        return_type="float"
    )
    
    # Verify response structure
    assert 'code' in response
    assert 'validation_results' in response
    assert response['validation_results']['syntax_valid']
    
    # Verify generated code
    code = response['code']
    assert 'def add_positive_numbers' in code
    assert 'float' in code
    assert 'raise ValueError' in code.lower()
    assert 'return' in code
    
    # Basic syntax check
    compile(code, '<string>', 'exec')
    
    # Execute the code to verify functionality
    namespace = {}
    exec(code, namespace)
    add_func = namespace['add_positive_numbers']
    
    # Test valid inputs
    assert add_func(5.0, 3.0) == 8.0
    
    # Test error cases
    with pytest.raises(ValueError):
        add_func(-1.0, 5.0)

@pytest.mark.integration
def test_string_processing_function(function_generator):
    """Test generating a string processing function."""
    response = function_generator.generate_function(
        requirements="Convert a string to title case and remove extra whitespace. Handle empty strings gracefully.",
        function_name="clean_and_title",
        parameters=[
            Parameter(name="text", type="str", description="Input text to process")
        ],
        return_type="str"
    )
    
    assert response['validation_results']['syntax_valid']
    assert 'def clean_and_title' in response['code']
    
    # Execute and test the function
    namespace = {}
    exec(response['code'], namespace)
    clean_func = namespace['clean_and_title']
    
    assert clean_func("hello  world") == "Hello World"
    assert clean_func("") == ""
    assert clean_func("  multiple   spaces  ") == "Multiple Spaces"

@pytest.mark.integration
def test_complex_function_generation(function_generator):
    """Test generating a more complex function with type hints."""
    response = function_generator.generate_function(
        requirements="""Filter and transform a list of dictionaries:
        1. Keep only items where the specified key exists and matches the value
        2. Transform matching items using the transform_func
        3. Maintain order of filtered items""",
        function_name="filter_and_transform",
        parameters=[
            Parameter(name="items", type="List[Dict[str, Any]]", description="List of dictionaries to filter"),
            Parameter(name="key", type="str", description="Key to filter on"),
            Parameter(name="value", type="Any", description="Value to match"),
            Parameter(name="transform_func", type="Callable[[Dict[str, Any]], Dict[str, Any]]", description="Function to transform matching items")
        ],
        return_type="List[Dict[str, Any]]"
    )
    
    assert response['validation_results']['syntax_valid']
    assert 'from typing import' in response['code']
    assert 'List[Dict[str, Any]]' in response['code']
    assert 'Callable' in response['code']
    
    # Execute and test the function
    namespace = {}
    exec(response['code'], namespace)
    filter_func = namespace['filter_and_transform']
    
    # Test data
    test_data = [
        {"id": 1, "name": "test1"},
        {"id": 2, "value": "keep"},
        {"id": 3, "value": "keep"},
        {"id": 4, "name": "test4"}
    ]
    
    def transform(item: Dict[str, Any]) -> Dict[str, Any]:
        return {**item, "transformed": True}
    
    result = filter_func(test_data, "value", "keep", transform)
    assert len(result) == 2
    assert all(item["transformed"] for item in result)
    assert all(item["value"] == "keep" for item in result)

@pytest.mark.integration
def test_error_handling_generation(function_generator):
    """Test generating a function with error handling."""
    response = function_generator.generate_function(
        requirements="""Parse a date string in multiple formats:
        1. Try ISO format first
        2. Then try common formats (YYYY-MM-DD, DD/MM/YYYY)
        3. Raise ValueError with helpful message if parsing fails""",
        function_name="parse_date",
        parameters=[
            Parameter(name="date_str", type="str", description="Date string to parse")
        ],
        return_type="datetime.datetime"
    )
    
    assert response['validation_results']['syntax_valid']
    assert 'import datetime' in response['code']
    assert 'ValueError' in response['code']
    assert 'try:' in response['code']
    assert 'except' in response['code']
    
    # Execute and test the function
    namespace = {}
    exec(response['code'], namespace)
    parse_func = namespace['parse_date']
    
    # Test valid dates
    assert isinstance(parse_func("2024-01-01"), namespace['datetime'].datetime)
    assert isinstance(parse_func("01/01/2024"), namespace['datetime'].datetime)
    
    # Test invalid date
    with pytest.raises(ValueError):
        parse_func("invalid date") 