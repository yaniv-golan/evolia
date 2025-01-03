"""Common prompts and templates used across code generation."""

# Base fields required in all responses
BASE_FIELDS = """1. "code": (string) The complete Python function code. Must be valid Python 3.
2. "function_name": (string) The exact function name from the requirements.
3. "parameters": (array) A list of parameter objects with "name", "type", "description".
4. "return_type": (string) The function's return type.
5. "validation_results": (object) containing:
   - "syntax_valid": (boolean) true if code is syntactically valid
   - "security_issues": (array of strings) any security concerns found
6. "outputs": (object) describing any outputs by name and type (optional or required as per your scenario).
7. "required_imports": (array) of strings listing any import statements needed by the function."""

# Chain of thought reasoning field
COT_FIELDS = """8. "cot_reasoning": (string) explaining how/why you wrote the code."""

# Common rules for all prompts
PROMPT_RULES = """Important Rules:
1. Return only this JSON object, *no extra text*.
2. The field "code" must have *complete*, *runnable* Python code.
3. Do not embed JSON in JSON. 
4. The function name, parameters, and return type must EXACTLY match what's requested.
5. Use only allowed modules and built-ins as specified.
6. Provide docstrings, type hints, and any error handling in the code.

If you cannot satisfy every rule, return an error in JSON."""

# Specialization text for different prompt types
FUNCTION_SPECIALIZATION = """You are a Python code generator that must return a single JSON object with these exact fields:"""

FIX_SPECIALIZATION = """You are a Python code fixer that must return a single JSON object with these exact fields.
You specialize in fixing Python code while:
1. Preserving original functionality
2. Maintaining existing interfaces
3. Fixing specific errors
4. Adding proper error handling"""

# Base validation schema
BASE_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "The complete Python code"},
        "validation_results": {
            "type": "object",
            "properties": {
                "syntax_valid": {"type": "boolean"},
                "security_issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["syntax_valid", "security_issues"],
            "additionalProperties": False,
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
                            "description": "The Python type of the output",
                        },
                        "reference": {
                            "type": "string",
                            "pattern": "^\\$[a-zA-Z_][a-zA-Z0-9_]*\\.[a-zA-Z_][a-zA-Z0-9_]*$",
                            "description": "Reference in format $stepname.outputname",
                        },
                    },
                    "required": ["type", "reference"],
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    },
    "required": ["code", "validation_results", "outputs"],
    "additionalProperties": False,
}


def create_prompt(prompt_type: str, include_cot: bool = False) -> str:
    """Create a system prompt for code generation or fixing.

    Args:
        prompt_type: Either 'function' or 'fix' to determine prompt type
        include_cot: Whether to include chain-of-thought fields

    Returns:
        Complete system prompt with appropriate fields and specialization
    """
    specialization = (
        FUNCTION_SPECIALIZATION
        if prompt_type == "function"
        else FIX_SPECIALIZATION
        if prompt_type == "fix"
        else FUNCTION_SPECIALIZATION  # Default to function
    )

    fields = f"{BASE_FIELDS}\n{COT_FIELDS}" if include_cot else BASE_FIELDS

    return f"""{specialization}

{fields}

{PROMPT_RULES}"""


# Create the actual prompts
FUNCTION_PROMPT = create_prompt("function", include_cot=False)
FUNCTION_PROMPT_WITH_COT = create_prompt("function", include_cot=True)
FIX_PROMPT = create_prompt("fix", include_cot=False)
FIX_PROMPT_WITH_COT = create_prompt("fix", include_cot=True)

# Chain of thought templates - only used with COT prompts
FUNCTION_COT_TEMPLATE = """Let's approach this step by step:
1. Understand the requirements
2. Verify the exact interface requirements:
   - Confirm the exact function name
   - Confirm the exact parameter names and types
   - Confirm the exact return type
3. Plan the implementation
4. Identify required imports
5. Consider edge cases
6. Write the implementation
7. Add error handling
8. Document with docstrings
9. Validate the interface matches exactly"""

FIX_COT_TEMPLATE = """Let's fix this step by step:
1. Analyze the error message
2. Identify the root cause
3. Consider edge cases
4. Implement the fix
5. Verify the solution
6. Add error handling"""
