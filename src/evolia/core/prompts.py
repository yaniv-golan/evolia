"""Common prompts and templates used across code generation."""

BASE_SYSTEM_PROMPT = """You are a Python code generator that follows these rules:
1. Generate complete, working Python code
2. Follow Python best practices and PEP-8
3. Include necessary imports
4. Use type hints
5. Add docstrings and comments
6. Handle edge cases and errors
7. Only use allowed modules and built-ins
8. Return values must match specified output types
9. Use output references in the format $stepname.outputname

Your response must include:
1. 'code': The complete Python function code
2. 'function_name': The name of the generated function
3. 'parameters': List of function parameters with name, type, and description
4. 'return_type': Function return type
5. 'validation_results': Object with syntax_valid and security_issues
6. 'outputs': Object mapping output names to types and descriptions"""

FUNCTION_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """
You specialize in generating Python functions that:
1. Have clear, descriptive names
2. Use appropriate parameters and return types
3. Include comprehensive docstrings
4. Handle all edge cases"""

FIX_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """
You specialize in fixing Python code while:
1. Preserving original functionality
2. Maintaining existing interfaces
3. Fixing specific errors
4. Adding proper error handling"""

# Chain of thought templates
FUNCTION_COT_TEMPLATE = """Let's approach this step by step:
1. Understand the requirements
2. Plan the function signature
3. Consider edge cases
4. Write the implementation
5. Add error handling
6. Document with docstrings"""

FIX_COT_TEMPLATE = """Let's fix this step by step:
1. Analyze the error message
2. Identify the root cause
3. Consider edge cases
4. Implement the fix
5. Verify the solution
6. Add error handling"""

# Base validation schema
BASE_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The complete Python code"
        },
        "validation_results": {
            "type": "object",
            "properties": {
                "syntax_valid": {"type": "boolean"},
                "security_issues": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["syntax_valid", "security_issues"]
        },
        "outputs": {
            "type": "object",
            "description": "Output definitions with types and references",
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "The Python type of the output"
                        },
                        "reference": {
                            "type": "string",
                            "pattern": "^\\$[a-zA-Z_][a-zA-Z0-9_]*\\.[a-zA-Z_][a-zA-Z0-9_]*$",
                            "description": "Reference in format $stepname.outputname"
                        }
                    },
                    "required": ["type", "reference"]
                }
            }
        }
    },
    "required": ["code", "validation_results", "outputs"]
} 