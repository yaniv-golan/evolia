"""Tests for logging functionality"""
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolia.utils.logger import (
    ColorFormatter,
    LogConfig,
    LogFormatter,
    LogHandler,
    code_generation_context,
    execution_context,
    setup_logger,
    validation_context,
)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing"""
    logger = MagicMock(spec=logging.Logger)
    return logger


def test_setup_logger_default():
    """Test logger setup with default settings"""
    logger = setup_logger()
    assert logger.name == "evolia"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # Console and file handler

    # Check console handler
    console_handler = next(
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    )
    assert console_handler.level == logging.INFO
    assert isinstance(console_handler.formatter, ColorFormatter)

    # Check file handler
    file_handler = next(
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.level == logging.DEBUG
    assert isinstance(file_handler.formatter, logging.Formatter)


def test_setup_logger_verbose():
    """Test logger setup with verbose mode"""
    logger = setup_logger(verbose=True)
    console_handler = next(
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    )
    assert console_handler.level == logging.DEBUG


def test_setup_logger_custom_file():
    """Test logger setup with custom log file"""
    test_log = Path("test.log")
    logger = setup_logger(log_file=test_log)
    file_handler = next(
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.baseFilename == str(test_log.absolute())


def test_color_formatter():
    """Test color formatter functionality"""
    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
    formatted = formatter.format(record)
    assert "INFO" in formatted
    assert "test message" in formatted


def test_color_formatter_with_payload():
    """Test color formatter with structured logging payload"""
    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
    record.payload = {"key": "value"}
    formatted = formatter.format(record)
    assert "test message" in formatted
    assert json.dumps({"key": "value"}, indent=2) in formatted


@pytest.mark.parametrize(
    "context_manager,name",
    [
        (code_generation_context, "test_step"),
        (validation_context, "test_step"),
        (execution_context, "test_step"),
    ],
)
def test_context_managers(context_manager, name, mock_logger):
    """Test logging context managers"""
    with patch("logging.getLogger", return_value=mock_logger):
        with context_manager(name):
            pass

        # Check that appropriate log messages were created
        assert mock_logger.info.called or mock_logger.debug.called


@pytest.mark.parametrize(
    "context_manager,name",
    [
        (code_generation_context, "test_step"),
        (validation_context, "test_step"),
        (execution_context, "test_step"),
    ],
)
def test_context_managers_with_error(context_manager, name, mock_logger):
    """Test logging context managers when an error occurs"""
    with patch("logging.getLogger", return_value=mock_logger):
        with pytest.raises(ValueError):
            with context_manager(name):
                raise ValueError("Test error")

        # Check that error was logged
        mock_logger.error.assert_called_once()
