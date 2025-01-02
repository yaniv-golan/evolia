"""Specialized function generator built on top of CodeGenerator."""

from typing import Dict, Any, Optional, List
import logging

from .code_generator import CodeGenerator
from .prompts import (
    FUNCTION_SYSTEM_PROMPT,
    FUNCTION_COT_TEMPLATE,
    BASE_VALIDATION_SCHEMA
)

logger = logging.getLogger(__name__)

class FunctionGenerator:
    """Specialized generator for Python functions."""
    
    def __init__(self, code_generator: CodeGenerator):
        """Initialize with CodeGenerator instance."""
        self.code_generator = code_generator
        
        # Template for generating functions
        self.function_template = """{cot_section}

Requirements:
{requirements}

The function MUST EXACTLY match this interface:
1. Name: {function_name}
2. Parameters (EXACT names and types required):
{parameters_structured}
3. Return type (EXACT type required): {return_type}

Additional requirements:
1. Only use these modules: {allowed_modules}
2. Only use these built-ins: {allowed_builtins}
3. Include proper error handling
4. Have comprehensive docstrings

Additional context:
{context}

Your response must include a 'cot_reasoning' field explaining your thought process."""

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
                                    "type": {"type": "string"}
                                }
                            }
                        },
                        "return_type": {"type": "string"},
                        "docstring": {"type": "string"}
                    },
                    "required": ["name", "parameters", "return_type", "docstring"]
                }
            },
            "required": ["code", "validation_results", "function_info"]
        }
        
        logger.debug("Initialized FunctionGenerator")

    def generate_function(
        self,
        requirements: str,
        function_name: str,
        parameters: List[Any],  # Can be List[Parameter] or List[Dict[str, str]]
        return_type: str,
        context: str = ""
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
        logger.debug("Generating function", extra={
            'payload': {
                'function_name': function_name,
                'parameters': parameters,
                'return_type': return_type
            }
        })
        
        # Format parameters for prompt
        params_structured = "\n".join([
            f"   - name: {p.name if hasattr(p, 'name') else p['name']}"
            f"\n     type: {p.type if hasattr(p, 'type') else p['type']}"
            for p in parameters
        ])
        
        # Prepare template variables
        template_vars = {
            'requirements': requirements,
            'function_name': function_name,
            'parameters_structured': params_structured,
            'return_type': return_type,
            'allowed_modules': ', '.join(sorted(self.code_generator.config.allowed_modules)),
            'allowed_builtins': ', '.join(sorted(self.code_generator.config.allowed_builtins)),
            'context': context,
            'cot_section': FUNCTION_COT_TEMPLATE if 'gpt-4' in self.code_generator.config.model.lower() else ""
        }
        
        try:
            # Generate function using code generator
            response = self.code_generator.generate(
                prompt_template=self.function_template,
                template_vars=template_vars,
                schema=self.function_schema,
                system_prompt=FUNCTION_SYSTEM_PROMPT
            )
            
            logger.debug("Function generated successfully", extra={
                'payload': {
                    'function_name': function_name,
                    'validation_results': response['validation_results']
                }
            })
            
            return response
            
        except Exception as e:
            logger.error("Function generation failed", extra={
                'payload': {
                    'function_name': function_name,
                    'error': str(e)
                }
            })
            raise 