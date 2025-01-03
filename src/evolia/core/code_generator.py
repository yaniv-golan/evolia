"""Core code generation functionality using OpenAI's structured output."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..core.prompts import BASE_VALIDATION_SCHEMA
from ..integrations.openai_structured import call_openai_structured
from ..models.models import Parameter
from ..models.schemas import CODE_SCHEMA
from ..utils.exceptions import APIRateLimitError, CodeGenerationError, TemplateError

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

    def _validate_template_vars(self, template: str, vars: Dict[str, Any]) -> None:
        """Validate template variables.

        Args:
            template: The template string
            vars: Dictionary of template variables

        Raises:
            TemplateError: If validation fails
        """
        # Extract required variables from template
        required_vars = {
            var.strip("{}") for var in re.findall(r"\{([^}]+)\}", template)
        }
        logger.debug(f"Required variables: {required_vars!r}")

        # Check for missing variables
        missing_vars = required_vars - vars.keys()
        logger.debug(f"Missing variables: {missing_vars!r}")
        if missing_vars:  # Raise error if there are any missing variables
            missing_var = next(iter(missing_vars))
            raise TemplateError(
                f"Missing required variable: {missing_var}",
                template=template,
                details={"missing_vars": list(missing_vars)},
            )

        # Check variable types
        for var_name, value in vars.items():
            logger.debug(f"Checking type of {var_name!r}: {type(value)}")
            # Allow certain non-string types for specific variables
            if var_name == "parameters" and isinstance(value, (list, tuple)):
                # Validate each parameter object
                for param in value:
                    if not isinstance(param, Parameter):
                        raise TemplateError(
                            f"Parameters must be Parameter objects: {param}",
                            template=template,
                            details={"var_name": var_name, "invalid_param": str(param)},
                        )
                # Convert parameters to string for template formatting
                vars[var_name] = ", ".join(str(param) for param in value)
            elif not isinstance(value, str):
                raise TemplateError(
                    f"Variable type must be string: {var_name}",
                    template=template,
                    details={"var_name": var_name, "actual_type": str(type(value))},
                )

        # Check for unknown variables
        unknown_vars = vars.keys() - required_vars
        logger.debug(f"Unknown variables: {unknown_vars!r}")
        if unknown_vars:  # Raise error if there are any unknown variables
            unknown_var = next(iter(unknown_vars))
            raise TemplateError(
                f"Unknown template variable: {unknown_var}",
                template=template,
                details={"unknown_vars": list(unknown_vars)},
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
            APIRateLimitError: If OpenAI API rate limit is exceeded
            TemplateError: If template validation fails
        """
        # Validate template variables first, before any API calls
        self._validate_template_vars(prompt_template, template_vars)

        # Format the prompt (this should now be safe since we validated the variables)
        prompt = prompt_template.format(**template_vars)

        try:
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

            # Handle empty response
            if not response:
                raise CodeGenerationError("Empty response")

            # Check for missing code
            if "code" not in response:
                raise CodeGenerationError("Missing code")

            # Check for null or empty code
            if response["code"] is None:
                raise CodeGenerationError("Null code")
            elif response["code"] == "":
                raise CodeGenerationError("Empty code")

            # Check for missing validation
            if (
                "validation_results" not in response
                or response["validation_results"] is None
            ):
                raise CodeGenerationError("Missing validation")

            # Check for syntax validation
            validation_results = response.get("validation_results", {})
            if not validation_results.get("syntax_valid", True) is True:
                raise CodeGenerationError("Generated code contains syntax errors")

            # Check for security issues
            security_issues = validation_results.get("security_issues", [])
            if security_issues:
                raise CodeGenerationError(
                    f"Security issues detected: {', '.join(security_issues)}"
                )

            # Validate parameter types
            valid_base_types = {
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "tuple",
                "set",
                "Any",
                "None",
                "Optional",
                "Union",
                "List",
                "Dict",
                "Set",
                "Tuple",
                "Callable",
                "Type",
                "Sequence",
                "Mapping",
                "Iterable",
                "Iterator",
                "Generator",
                "Coroutine",
                "AsyncIterator",
                "AsyncIterable",
                "Collection",
                "Container",
                "Sized",
                "ByteString",
                "Match",
                "Pattern",
                "TextIO",
                "BinaryIO",
                "IO",
            }

            def is_valid_type(type_str: str) -> bool:
                """Check if a type string is valid."""
                # Handle parameterized types like List[int], Dict[str, Any]
                if "[" in type_str and type_str.endswith("]"):
                    base_type = type_str.split("[")[0]
                    return base_type in valid_base_types
                # Handle Union types written with pipe
                if "|" in type_str:
                    types = [t.strip() for t in type_str.split("|")]
                    return all(t in valid_base_types for t in types)
                return type_str in valid_base_types

            parameters = response.get("parameters", [])
            for param in parameters:
                if not isinstance(param, dict) or "type" not in param:
                    raise CodeGenerationError("Invalid parameter format")
                param_type = param["type"]
                if not is_valid_type(param_type):
                    raise CodeGenerationError(f"Invalid parameter type: {param_type}")

            # Post-process response to ensure all required fields
            processed_response = {
                "code": response["code"],
                "function_name": response.get(
                    "function_name",
                    template_vars.get("function_name", "auto_generated"),
                ),
                "parameters": parameters,
                "return_type": response.get("return_type", "Any"),
                "description": response.get("description", ""),
                "validation_results": validation_results,
                "outputs": response.get("outputs", {}),
                "required_imports": response.get("required_imports", []),
            }

            return processed_response

        except (CodeGenerationError, APIRateLimitError, TemplateError):
            raise
        except Exception as e:
            error_msg = str(e)
            # Map error messages to expected phrases
            error_mappings = {
                "Invalid API key": "Authentication failed",
                "Model overloaded": "Server overload error",
                "Bad gateway": "Service unavailable",
                "Context length exceeded": "Input too long",
            }

            for pattern, mapped_msg in error_mappings.items():
                if pattern.lower() in error_msg.lower():
                    raise CodeGenerationError(mapped_msg)

            # If no mapping found, use original message
            raise CodeGenerationError(f"Failed to generate code: {error_msg}")
