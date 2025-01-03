"""JSON schemas for code generation and validation."""

# JSON Schema for OpenAI code generation
CODE_SCHEMA = {
    "type": "object",
    "description": "A Python function definition with metadata",
    "properties": {
        "code": {
            "type": "string",
            "description": "The complete Python function code. Must contain ONLY the function definition.",
            "minLength": 1,
        },
        "function_name": {
            "type": "string",
            "description": "The name of the generated function",
            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
        },
        "parameters": {
            "type": "array",
            "description": "List of function parameters",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Parameter name",
                        "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                    },
                    "type": {"type": "string", "description": "Parameter type"},
                    "description": {
                        "type": "string",
                        "description": "Description of the parameter",
                    },
                },
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        },
        "return_type": {"type": "string", "description": "Function return type"},
        "description": {
            "type": "string",
            "description": "Clear description of the function's purpose",
        },
        "required_imports": {
            "type": "array",
            "description": "List of import statements needed by the function (both for type hints and function body)",
            "items": {
                "type": "string",
                "description": "Complete import statement (e.g. 'from typing import List' or 'import re')",
            },
        },
        "validation_results": {
            "type": "object",
            "properties": {
                "syntax_valid": {"type": "boolean"},
                "security_issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["syntax_valid", "security_issues"],
        },
        "outputs": {
            "type": "object",
            "description": "Output definitions with types and descriptions",
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "The Python type of the output",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the output",
                        },
                    },
                    "required": ["type", "description"],
                }
            },
        },
    },
    "required": [
        "code",
        "function_name",
        "parameters",
        "return_type",
        "validation_results",
        "outputs",
        "required_imports",
    ],
}

"""Schema definitions for data validation."""

from typing import Dict, List, Optional, Union

# Base schemas
ParameterSchema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "description": {"type": "string"},
        "optional": {"type": "boolean"},
        "default": {"type": ["string", "number", "boolean", "null"]},
    },
    "required": ["name", "type", "description"],
    "additionalProperties": False,
}

ReturnTypeSchema = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "description": {"type": "string"},
        "schema": {"type": "object"},  # For complex return types
    },
    "required": ["type"],
    "additionalProperties": False,
}

# Function schema
FunctionSchema = {
    "type": "object",
    "properties": {
        "function_name": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": ParameterSchema,
        },
        "return_type": {"type": "string"},
        "description": {"type": "string"},
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["function_name", "parameters", "return_type"],
    "additionalProperties": False,
}

# Validation result schema
ValidationResultSchema = {
    "type": "object",
    "properties": {
        "syntax_valid": {"type": "boolean"},
        "security_issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "type_issues": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    },
    "required": ["syntax_valid", "security_issues"],
    "additionalProperties": False,
}
