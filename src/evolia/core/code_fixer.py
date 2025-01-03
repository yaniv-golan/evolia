"""Specialized code fixer built on top of CodeGenerator."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .code_generator import CodeGenerator
from .prompts import BASE_VALIDATION_SCHEMA, FIX_COT_TEMPLATE, FIX_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class CodeFixConfig:
    """Configuration specific to code fixing."""

    fix_temperature: float = 0.1
    max_attempts: int = 3

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.fix_temperature <= 1:
            raise ValueError("fix_temperature must be between 0 and 1")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0")


@dataclass
class FixAttempt:
    """Record of a fix attempt."""

    code: str
    error: str
    fixed_code: str
    timestamp: datetime = field(default_factory=datetime.now)
    validation_results: Dict[str, Any] = field(default_factory=dict)


class FixHistory:
    """Tracks the history of fix attempts."""

    def __init__(self):
        self.attempts: List[FixAttempt] = []

    def add_attempt(self, code: str, error: str, response: Dict[str, Any]):
        """Add a fix attempt to history."""
        self.attempts.append(
            FixAttempt(
                code=code,
                error=error,
                fixed_code=response["code"],
                validation_results=response["validation_results"],
            )
        )

    def get_previous_fixes(self) -> List[str]:
        """Get list of previously attempted fixes."""
        return [attempt.fixed_code for attempt in self.attempts]


class CodeFixer:
    """Specialized code fixer using CodeGenerator."""

    def __init__(
        self, code_generator: CodeGenerator, config: Optional[CodeFixConfig] = None
    ):
        """Initialize code fixer with CodeGenerator instance."""
        self.code_generator = code_generator
        self.config = config or CodeFixConfig()
        self.fix_history = FixHistory()

        # Template for fixing code
        self.fix_template = """{cot_section}

Original Code:
{original_code}

Error Message:
{error_msg}

Previous Fix Attempts:
{previous_fixes}

The fixed code must:
1. Fix the specific error while preserving the original functionality
2. Keep the exact same function signature
3. Only use these modules: {allowed_modules}
4. Only use these built-ins: {allowed_builtins}
5. Handle the specific error case that occurred

Your response must include a 'cot_reasoning' field explaining your thought process."""

        # Schema for fix response
        self.fix_schema = {
            **BASE_VALIDATION_SCHEMA,
            "properties": {
                **BASE_VALIDATION_SCHEMA["properties"],
                "fix_description": {
                    "type": "string",
                    "description": "Brief explanation of the fix",
                },
            },
            "required": ["code", "validation_results", "fix_description"],
        }

        logger.debug(
            "Initialized CodeFixer",
            extra={
                "payload": {
                    "fix_temperature": self.config.fix_temperature,
                    "max_attempts": self.config.max_attempts,
                }
            },
        )

    def fix_code(self, code: str, error_msg: str, attempt: int = 0) -> Dict[str, Any]:
        """
        Fix code based on error message.

        Args:
            code: Original code to fix
            error_msg: Error message to fix
            attempt: Current attempt number

        Returns:
            Dict containing fixed code and validation results
        """
        if attempt >= self.config.max_attempts:
            logger.error(
                "Maximum fix attempts reached",
                extra={
                    "payload": {
                        "max_attempts": self.config.max_attempts,
                        "error_msg": error_msg,
                    }
                },
            )
            raise ValueError(
                f"Failed to fix code after {self.config.max_attempts} attempts"
            )

        logger.debug(
            "Attempting to fix code",
            extra={"payload": {"attempt": attempt, "error_msg": error_msg}},
        )

        # Get previous fixes for context
        previous_fixes = self.fix_history.get_previous_fixes()
        previous_fixes_str = (
            "\n".join(
                [f"Attempt {i+1}:\n{fix}" for i, fix in enumerate(previous_fixes)]
            )
            if previous_fixes
            else "None"
        )

        # Check if using GPT-4 for COT
        model_name = self.code_generator.config.model.lower()
        cot_section = FIX_COT_TEMPLATE if "gpt-4" in model_name else ""

        # Prepare template variables
        template_vars = {
            "original_code": code,
            "error_msg": error_msg,
            "previous_fixes": previous_fixes_str,
            "allowed_modules": ", ".join(
                sorted(self.code_generator.config.allowed_modules)
            ),
            "allowed_builtins": ", ".join(
                sorted(self.code_generator.config.allowed_builtins)
            ),
            "cot_section": cot_section,
        }

        try:
            # Generate fix using code generator
            response = self.code_generator.generate(
                prompt_template=self.fix_template,
                template_vars=template_vars,
                schema=self.fix_schema,
                system_prompt=FIX_SYSTEM_PROMPT,
            )

            # Add to fix history
            self.fix_history.add_attempt(code, error_msg, response)

            logger.debug(
                "Fix generated successfully",
                extra={
                    "payload": {
                        "attempt": attempt,
                        "validation_results": response["validation_results"],
                    }
                },
            )

            return response

        except Exception as e:
            logger.error(
                "Fix generation failed",
                extra={"payload": {"attempt": attempt, "error": str(e)}},
            )
            # Retry with next attempt
            return self.fix_code(code, error_msg, attempt + 1)
