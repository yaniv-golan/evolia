import os
import json
import logging
import argparse
from typing import Dict, Any
from pathlib import Path

from evolia.models.models import CodeGenerationRequest, Parameter
from evolia.integrations.openai_structured import call_openai_structured

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# JSON Schema for OpenAI code generation (copied from executor.py)
CODE_SCHEMA = {
    "type": "object",
    "description": "A Python function definition with metadata",
    "properties": {
        "code": {
            "type": "string",
            "description": "The complete Python function code. Must contain ONLY the function definition.",
            "minLength": 1
        },
        "function_name": {
            "type": "string",
            "description": "The name of the generated function",
            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"
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
                        "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"
                    },
                    "type": {
                        "type": "string",
                        "description": "Parameter type"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the parameter"
                    }
                },
                "required": ["name", "type"],
                "additionalProperties": False
            }
        },
        "return_type": {
            "type": "string",
            "description": "Function return type",
            "enum": ["dict"]
        },
        "description": {
            "type": "string",
            "description": "Clear description of the function's purpose"
        },
        "examples": {
            "type": "array",
            "description": "Example usages of the function (optional)",
            "items": {
                "type": "string"
            }
        },
        "constraints": {
            "type": "array",
            "description": "Any constraints or special conditions for the function (optional)",
            "items": {
                "type": "string"
            }
        },
        "cot_reasoning": {
            "type": "string",
            "description": "Chain-of-thought reasoning explaining how the code was derived. NO CODE HERE!"
        },
        "validation_results": {
            "type": "object",
            "description": "Results of code validation",
            "properties": {
                "syntax_valid": {
                    "type": "boolean",
                    "description": "Whether the code is syntactically valid"
                },
                "name_matches": {
                    "type": "boolean",
                    "description": "Whether the function name matches requirements"
                },
                "params_match": {
                    "type": "boolean",
                    "description": "Whether the parameters match requirements"
                },
                "return_type_matches": {
                    "type": "boolean",
                    "description": "Whether the return type matches requirements"
                },
                "security_issues": {
                    "type": "array",
                    "description": "List of security issues found",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["syntax_valid", "name_matches", "params_match", "return_type_matches", "security_issues"],
            "additionalProperties": False
        }
    },
    "required": ["code", "function_name", "parameters", "return_type", "description", "cot_reasoning", "validation_results"],
    "additionalProperties": False
}

def format_json(obj: Dict[str, Any], indent: int = 2) -> str:
    """Format a JSON object with proper indentation and line breaks."""
    return json.dumps(obj, indent=indent)

def format_request(messages: list, model: str) -> str:
    """Format the OpenAI request in a readable manner."""
    formatted_request = [
        "=== OpenAI Request ===",
        f"\nModel: {model}",
        "\nSystem Prompt:",
        "--------------",
        messages[0]['content'],
        "\nUser Prompt:",
        "------------",
        messages[1]['content']
    ]
    return "\n".join(formatted_request)

def format_response(response: Dict[str, Any]) -> str:
    """Format the OpenAI response in a readable manner."""
    formatted_sections = [
        "\n=== OpenAI Response ===",
        "\n1. Generated Code:",
        "------------------",
        response["code"],
        "\n2. Function Metadata:",
        "-------------------",
        f"Name: {response['function_name']}",
        "\nParameters:",
    ]
    
    # Format parameters
    for param in response["parameters"]:
        formatted_sections.extend([
            f"  - {param['name']}: {param['type']}",
            f"    Description: {param['description']}"
        ])
    
    formatted_sections.extend([
        f"\nReturn Type: {response['return_type']}",
        "\n3. Function Description:",
        "----------------------",
        response["description"],
        "\n4. Chain of Thought:",
        "------------------",
        response["cot_reasoning"],
        "\n5. Validation Results:",
        "--------------------"
    ])
    
    # Format validation results
    validation = response["validation_results"]
    formatted_sections.extend([
        f"Syntax Valid: {validation['syntax_valid']}",
        f"Name Matches: {validation['name_matches']}",
        f"Parameters Match: {validation['params_match']}",
        f"Return Type Matches: {validation['return_type_matches']}"
    ])
    
    if validation["security_issues"]:
        formatted_sections.append("\nSecurity Issues:")
        for issue in validation["security_issues"]:
            formatted_sections.append(f"  - {issue}")
    else:
        formatted_sections.append("\nNo security issues found.")
    
    # Add raw JSON response
    formatted_sections.extend([
        "\n=== Raw JSON Response ===",
        "----------------------",
        json.dumps({
            "code": response["code"],
            "function_name": response["function_name"],
            "parameters": response["parameters"],
            "return_type": response["return_type"],
            "description": response["description"],
            "cot_reasoning": response["cot_reasoning"],
            "validation_results": response["validation_results"]
        }, indent=2)
    ])
    
    return "\n".join(formatted_sections)

def test_code_generation(description: str):
    """Test code generation with a given description."""
    # Create a test request
    request = CodeGenerationRequest(
        function_name="process_number",
        parameters=[
            Parameter(name="url", type="str", description="URL of the content to fetch"),
            Parameter(name="output_file_path", type="str", description="Path to save the fetched content")
        ],
        return_type="dict",
        constraints=["no_globals", "no_nested_functions"],
        description=description
    )
    
    # Prepare request messages
    system_prompt = """You are a Python code generator that creates clean, efficient functions.
Your response must be a valid JSON object, but before generating it, you must:

1. Generate the function code according to the requirements
2. Perform these validation checks:
   a. Syntax validation:
      - Check if the code is valid Python syntax
      - Verify all required imports are included
      - Ensure proper indentation and formatting
   b. Name validation:
      - Compare the function name with the requested name
      - Verify it follows Python naming conventions
   c. Parameter validation:
      - Check if all required parameters are present
      - Verify parameter types match the requirements
      - Ensure parameter names are valid Python identifiers
   d. Return type validation:
      - Verify the return type annotation matches requirements
      - Check if the actual return values match the type
   e. Security validation:
      - Check for potential security issues
      - Look for unsafe operations
      - Identify any risky code patterns

3. Include your validation process in the chain-of-thought reasoning, explaining:
   - How you performed each validation check
   - What issues you found (if any)
   - How you resolved any issues

Then return a JSON object with:
1. The exact function code as specified
2. The function name exactly as requested
3. The parameter list with types
4. The return type annotation
5. A clear description of what the function does
6. Chain-of-thought reasoning including your validation process
7. Validation results based on your actual checks

The code must be syntactically correct Python and follow all constraints."""

    user_prompt = f"""Generate a Python function that {request.description}

The response must be a JSON object with these fields:
1. "code": The complete function definition (ONLY the function, no other code)
2. "function_name": {request.function_name}
3. "parameters": {request.parameters}
4. "return_type": "dict"
5. "description": A clear description of what the function does
6. "cot_reasoning": Your step-by-step explanation of how you created this code AND how you validated it
   - Explain your thought process for both implementation and validation
   - Show your validation checks and results
   - NO code snippets here
   - Focus on high-level reasoning and validation process
7. "validation_results": {{
    "syntax_valid": <result of your syntax check>,
    "name_matches": <result of your name validation>,
    "params_match": <result of your parameter validation>,
    "return_type_matches": <result of your return type validation>,
    "security_issues": [<list of any security issues found>]
}}

Return ONLY this JSON object. No other text or code."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Print formatted request
    print(format_request(messages, "gpt-4o-2024-08-06"))
    
    # Run the test
    response = call_openai_structured(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-2024-08-06",  # Using the same model as the main system
        json_schema=CODE_SCHEMA,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_retries=3,
        retry_delay=20
    )
    
    # Print formatted response
    print(format_response(response))
    
    assert response is not None
    assert "code" in response
    assert "function_name" in response
    assert "parameters" in response
    assert "return_type" in response
    assert "description" in response
    assert "cot_reasoning" in response
    assert "validation_results" in response

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test OpenAI code generation with a given task description")
    parser.add_argument("description", help="Description of the code to generate (e.g., 'Read a number from input file, double it, add 10, and write the result to output file')")
    args = parser.parse_args()

    # Configure logging to hide debug messages
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('evolia').setLevel(logging.WARNING)
    
    test_code_generation(args.description) 