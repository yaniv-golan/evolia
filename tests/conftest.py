import pytest
from pathlib import Path

@pytest.fixture
def config():
    """Test configuration with mock endpoints"""
    return {
        "hugging_face": {
            "instruct_endpoint_url": "https://mock_instruct_url",
            "python_endpoint_url": "https://mock_python_url",
            "max_retries": 5,
            "retry_delay": 1
        },
        "allowed_modules": ["os", "pathlib", "json"],
        "max_runtime_retries": 2,
        "max_syntax_lint_retries": 3
    }

@pytest.fixture
def endpoint_url():
    """Mock endpoint URL for local LLM tests"""
    return "https://mock_endpoint_url"

@pytest.fixture
def clean_environment(tmp_path):
    """Create a clean test environment"""
    original_cwd = Path.cwd()
    Path.cwd().joinpath("run_artifacts").mkdir(exist_ok=True)
    yield tmp_path
    # Cleanup after test
    if Path("run_artifacts").exists():
        import shutil
        shutil.rmtree("run_artifacts")

@pytest.fixture
def mock_openai_config():
    """Test configuration for OpenAI with mock API key"""
    return {
        "api_key": "test-key",
        "model": "gpt-4o-2024-08-06",
        "temperature": 0.5,
        "max_tokens": 100,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "allowed_modules": ["math", "typing"],
        "allowed_builtins": ["len", "str", "int", "float"]
    }

@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock OpenAI client for testing"""
    from unittest.mock import Mock
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response", function_call=None))]
    )
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)
    return mock_client
  