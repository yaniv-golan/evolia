"""Core code generation functionality using OpenAI's structured output."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..core.prompts import BASE_VALIDATION_SCHEMA
from ..integrations.openai_structured import call_openai_structured
from ..models.models import Parameter
from ..models.schemas import CODE_SCHEMA
from ..utils.exceptions import CodeGenerationError

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationConfig:
    """Configuration for code generation."""

    api_key: str
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.2
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    allowed_modules: Set[str] = field(
        default_factory=lambda: {
            "typing",
            "datetime",
            "json",
            "logging",
            "re",
            "math",
            "collections",
            "itertools",
            "functools",
        }
    )
    allowed_builtins: Set[str] = field(
        default_factory=lambda: {
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "len",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "min",
            "max",
            "sum",
            "any",
            "all",
            "isinstance",
            "hasattr",
        }
    )

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if not -2 <= self.frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")


class CodeGenerator:
    """Core code generation using OpenAI's structured output."""

    def __init__(self, config: Optional[CodeGenerationConfig] = None):
        """Initialize with configuration."""
        if config is None:
            raise ValueError("Configuration must be provided with API key")
        self.config = config

        logger.debug(
            "Initialized CodeGenerator",
            extra={
                "payload": {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "allowed_modules": sorted(self.config.allowed_modules),
                    "allowed_builtins": sorted(self.config.allowed_builtins),
                }
            },
        )

    def generate(
        self,
        prompt_template: str,
        template_vars: Dict[str, Any],
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate code using OpenAI API.

        Args:
            prompt_template: Template for the prompt
            template_vars: Variables to fill in the template
            schema: JSON schema for response validation
            system_prompt: Optional system prompt

        Returns:
            Dict containing the generated code and metadata

        Raises:
            CodeGenerationError: If code generation fails
        """
        try:
            # Format the prompt
            prompt = prompt_template.format(**template_vars)

            # Make OpenAI API call
            response = call_openai_structured(
                api_key=self.config.api_key,
                model=self.config.model,
                json_schema=schema,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
            )

            # Post-process response to ensure all required fields
            processed_response = {
                "code": response["code"],
                "function_name": response.get(
                    "function_name",
                    template_vars.get("function_name", "auto_generated"),
                ),
                "parameters": response.get("parameters", []),
                "return_type": response.get("return_type", "Any"),
                "description": response.get("description", ""),
                "validation_results": response.get(
                    "validation_results", {"security_issues": []}
                ),
                "outputs": response.get("outputs", {}),
                "required_imports": response.get("required_imports", []),
            }

            return processed_response

        except Exception as e:
            raise CodeGenerationError(f"Failed to generate code: {str(e)}")
