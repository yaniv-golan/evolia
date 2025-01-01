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
  