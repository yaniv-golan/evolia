"""Unit tests for CodeFixer."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from evolia.core.code_fixer import CodeFixConfig, CodeFixer, FixAttempt, FixHistory
from evolia.core.code_generator import CodeGenerationConfig, CodeGenerator


@pytest.fixture
def mock_code_generator():
    """Create a mock CodeGenerator."""
    generator = Mock(spec=CodeGenerator)
    # Create a real config object for the mock
    generator.config = CodeGenerationConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=100,
        allowed_modules={"math", "typing"},
        allowed_builtins={"len", "str"},
        api_key="test-key",
    )
    generator.generate.return_value = {
        "code": "def fixed(): pass",
        "validation_results": {"syntax_valid": True, "security_issues": []},
        "fix_description": "Fixed the issue",
    }
    return generator


@pytest.fixture
def code_fixer(mock_code_generator):
    """Create a CodeFixer instance with test config."""
    config = CodeFixConfig(fix_temperature=0.2, max_attempts=2)
    return CodeFixer(mock_code_generator, config)


def test_code_fixer_init_default_config(mock_code_generator):
    """Test CodeFixer initialization with default config."""
    fixer = CodeFixer(mock_code_generator)
    assert fixer.config.fix_temperature == 0.1
    assert fixer.config.max_attempts == 3
    assert isinstance(fixer.fix_history, FixHistory)


def test_code_fixer_init_custom_config(mock_code_generator):
    """Test CodeFixer initialization with custom config."""
    config = CodeFixConfig(fix_temperature=0.2, max_attempts=5)
    fixer = CodeFixer(mock_code_generator, config)
    assert fixer.config.fix_temperature == 0.2
    assert fixer.config.max_attempts == 5


def test_code_fix_config_validation():
    """Test CodeFixConfig validation."""
    with pytest.raises(ValueError):
        CodeFixConfig(fix_temperature=1.5)

    with pytest.raises(ValueError):
        CodeFixConfig(max_attempts=0)


def test_fix_attempt_creation():
    """Test FixAttempt creation and defaults."""
    attempt = FixAttempt(
        code="def test(): pass", error="NameError", fixed_code="def test(): return None"
    )
    assert attempt.code == "def test(): pass"
    assert attempt.error == "NameError"
    assert attempt.fixed_code == "def test(): return None"
    assert isinstance(attempt.timestamp, datetime)
    assert isinstance(attempt.validation_results, dict)


def test_fix_history_tracking():
    """Test FixHistory tracking functionality."""
    history = FixHistory()

    # Add first attempt
    history.add_attempt(
        code="def test(): pass",
        error="NameError",
        response={
            "code": "def test(): return None",
            "validation_results": {"syntax_valid": True},
        },
    )

    # Add second attempt
    history.add_attempt(
        code="def test(): return None",
        error="TypeError",
        response={
            "code": "def test() -> None: return None",
            "validation_results": {"syntax_valid": True},
        },
    )

    previous_fixes = history.get_previous_fixes()
    assert len(previous_fixes) == 2
    assert previous_fixes[0] == "def test(): return None"
    assert previous_fixes[1] == "def test() -> None: return None"


def test_fix_code_success(code_fixer):
    """Test successful code fixing."""
    response = code_fixer.fix_code(
        code="def test(): pass", error_msg="NameError: name 'x' is not defined"
    )

    assert response["code"] == "def fixed(): pass"
    assert response["validation_results"]["syntax_valid"]
    assert response["fix_description"] == "Fixed the issue"

    # Verify the fix was added to history
    assert len(code_fixer.fix_history.attempts) == 1


def test_fix_code_max_attempts(code_fixer):
    """Test fix_code reaches max attempts."""
    code_fixer.code_generator.generate.side_effect = Exception("Test error")

    with pytest.raises(ValueError) as exc_info:
        code_fixer.fix_code(code="def test(): pass", error_msg="Error")

    assert "Failed to fix code after 2 attempts" in str(exc_info.value)
    assert len(code_fixer.fix_history.attempts) == 0


def test_fix_code_retry_success(code_fixer):
    """Test fix_code succeeds on retry."""

    def generate_side_effect(*args, **kwargs):
        if generate_side_effect.calls == 0:
            generate_side_effect.calls += 1
            raise Exception("First attempt failed")
        return {
            "code": "def fixed(): pass",
            "validation_results": {"syntax_valid": True},
            "fix_description": "Fixed on retry",
        }

    generate_side_effect.calls = 0
    code_fixer.code_generator.generate.side_effect = generate_side_effect

    response = code_fixer.fix_code(code="def test(): pass", error_msg="Error")

    assert response["code"] == "def fixed(): pass"
    assert response["fix_description"] == "Fixed on retry"
    assert len(code_fixer.fix_history.attempts) == 1


def test_fix_code_with_previous_attempts(code_fixer):
    """Test fix_code includes previous attempts in prompt."""
    # Add a previous attempt
    code_fixer.fix_history.add_attempt(
        code="def test(): pass",
        error="NameError",
        response={
            "code": "def test(): return None",
            "validation_results": {"syntax_valid": True},
        },
    )

    # Fix code again
    code_fixer.fix_code(code="def test(): return None", error_msg="TypeError")

    # Verify previous attempt was included in prompt
    call_args = code_fixer.code_generator.generate.call_args[1]
    template_vars = call_args["template_vars"]
    assert "def test(): return None" in template_vars["previous_fixes"]
