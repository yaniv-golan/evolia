"""Specialized function generator built on top of CodeGenerator."""

import logging
from typing import Any, Dict, List, Optional

from .code_generator import CodeGenerator
from .prompts import (
    BASE_VALIDATION_SCHEMA,
    FUNCTION_COT_TEMPLATE,
    FUNCTION_PROMPT,
    FUNCTION_PROMPT_WITH_COT,
)

logger = logging.getLogger(__name__)


class FunctionGenerator:
    """Function generator that uses CodeGenerator."""

    def __init__(self, code_generator: CodeGenerator):
        """Initialize with CodeGenerator instance."""
        self.code_generator = code_generator

        # Template for generating functions
        self.function_template = """{cot_section}

Requirements: 
1. Write a function "{function_name}" that:
   - Has parameters {parameters_structured}
   - Returns a {return_type}
2. The function must do exactly:
   {requirements}

Allowed modules: {allowed_modules}
Allowed built-ins: {allowed_builtins}

Additional requirements:
- Must have a docstring describing the function
- Must handle errors gracefully
{context}

Return *only* the JSON object with the fields described in the system prompt."""

        # Schema for function generation
        self.function_schema = {
            **BASE_VALIDATION_SCHEMA,
            "properties": {
                **BASE_VALIDATION_SCHEMA["properties"],
                "function_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                            },
                        },
                        "return_type": {"type": "string"},
                        "docstring": {"type": "string"},
                    },
                    "required": ["name", "parameters", "return_type", "docstring"],
                },
            },
            "required": ["code", "validation_results", "function_info"],
        }

        logger.debug("Initialized FunctionGenerator")

    def generate_function(
        self,
        requirements: str,
        function_name: str,
        parameters: List[Any],  # Can be List[Parameter] or List[Dict[str, str]]
        return_type: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a Python function based on requirements.

        Args:
            requirements: Description of what the function should do
            function_name: Name of the function to generate
            parameters: List of Parameter objects or parameter dictionaries with name and type
            return_type: Return type of the function
            context: Additional context or constraints

        Returns:
            Dict containing generated code and metadata
        """
        logger.debug(
            "Generating function",
            extra={
                "payload": {
                    "function_name": function_name,
                    "parameters": parameters,
                    "return_type": return_type,
                }
            },
        )

        # Format parameters for prompt
        params_structured = "\n".join(
            [
                f"   - name: {p.name if hasattr(p, 'name') else p['name']}"
                f"\n     type: {p.type if hasattr(p, 'type') else p['type']}"
                for p in parameters
            ]
        )

        # Check if using GPT-4 for COT
        model_name = self.code_generator.config.model.lower()
        is_gpt4 = "gpt-4" in model_name
        cot_section = FUNCTION_COT_TEMPLATE if is_gpt4 else ""

        # Add cot_reasoning to required fields if using GPT-4
        schema = self.function_schema.copy()
        if is_gpt4:
            schema["required"] = list(schema.get("required", [])) + ["cot_reasoning"]

        # Prepare template variables
        template_vars = {
            "requirements": requirements,
            "function_name": function_name,
            "parameters_structured": params_structured,
            "return_type": return_type,
            "allowed_modules": ", ".join(
                sorted(self.code_generator.config.allowed_modules)
            ),
            "allowed_builtins": ", ".join(
                sorted(self.code_generator.config.allowed_builtins)
            ),
            "context": context,
            "cot_section": cot_section,
        }

        try:
            # Generate function using code generator
            response = self.code_generator.generate(
                prompt_template=self.function_template,
                template_vars=template_vars,
                schema=schema,
                system_prompt=FUNCTION_PROMPT_WITH_COT if is_gpt4 else FUNCTION_PROMPT,
            )

            logger.debug(
                "Function generated successfully",
                extra={
                    "payload": {
                        "function_name": function_name,
                        "validation_results": response["validation_results"],
                    }
                },
            )

            return response

        except Exception as e:
            logger.error(
                "Function generation failed",
                extra={"payload": {"function_name": function_name, "error": str(e)}},
            )
            raise
