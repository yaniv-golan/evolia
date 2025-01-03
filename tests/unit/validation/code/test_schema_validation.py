"""Tests for schema validation functionality."""
import pytest

from evolia.models.schemas import (
    FunctionSchema,
    ParameterSchema,
    ReturnTypeSchema,
    ValidationResultSchema,
)
from evolia.utils.exceptions import ValidationError
from evolia.validation.code_validation import validate_schema


@pytest.fixture
def valid_function_schema():
    """Create a valid function schema for testing."""
    return {
        "function_name": "test_function",
        "parameters": [
            {"name": "param1", "type": "int", "description": "First parameter"}
        ],
        "return_type": "str",
        "description": "Test function",
    }


def test_validate_function_schema(valid_function_schema):
    """Test validation of a valid function schema."""
    result = validate_schema(valid_function_schema, FunctionSchema)
    assert result is not None
    assert result["function_name"] == "test_function"
    assert len(result["parameters"]) == 1


def test_validate_invalid_function_schema():
    """Test validation of an invalid function schema."""
    invalid_schema = {
        "function_name": 123,  # Should be string
        "parameters": "not_a_list",  # Should be list
        "return_type": "str",
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(invalid_schema, FunctionSchema)
    assert "function_name" in str(exc_info.value)
    assert "parameters" in str(exc_info.value)


def test_validate_parameter_schema():
    """Test validation of parameter schema."""
    valid_param = {"name": "test_param", "type": "int", "description": "Test parameter"}
    result = validate_schema(valid_param, ParameterSchema)
    assert result is not None
    assert result["name"] == "test_param"
    assert result["type"] == "int"


def test_validate_return_type_schema():
    """Test validation of return type schema."""
    valid_return = {"type": "list", "description": "Returns a list"}
    result = validate_schema(valid_return, ReturnTypeSchema)
    assert result is not None
    assert result["type"] == "list"


def test_validate_validation_result_schema():
    """Test validation of validation result schema."""
    valid_result = {"syntax_valid": True, "security_issues": [], "type_issues": None}
    result = validate_schema(valid_result, ValidationResultSchema)
    assert result is not None
    assert result["syntax_valid"] is True
    assert len(result["security_issues"]) == 0


def test_validate_missing_required_fields():
    """Test validation when required fields are missing."""
    invalid_schema = {"parameters": []}  # Missing function_name and return_type
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(invalid_schema, FunctionSchema)
    assert "function_name" in str(exc_info.value)
    assert "return_type" in str(exc_info.value)


def test_validate_additional_properties():
    """Test validation with additional properties."""
    schema_with_extra = {
        "function_name": "test",
        "parameters": [],
        "return_type": "void",
        "extra_field": "should not be here",
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(schema_with_extra, FunctionSchema)
    assert "extra_field" in str(exc_info.value)


def test_validate_nested_schema():
    """Test validation of nested schema structures."""
    nested_schema = {
        "function_name": "complex_function",
        "parameters": [
            {
                "name": "nested_param",
                "type": "dict",
                "description": "A nested parameter",
                "schema": {
                    "type": "object",
                    "properties": {
                        "field1": {"type": "string"},
                        "field2": {"type": "integer"},
                    },
                },
            }
        ],
        "return_type": "dict",
        "description": "Function with nested schema",
    }
    result = validate_schema(nested_schema, FunctionSchema)
    assert result is not None
    assert result["parameters"][0]["type"] == "dict"
    assert "schema" in result["parameters"][0]
