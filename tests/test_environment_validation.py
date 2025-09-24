"""
Tests for environment variable validation at startup.
"""

import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from src.cli import _validate_environment, app


def test_github_token_validation_valid():
    """Test that valid GitHub token passes validation."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "valid_token_123"}):
        # Should not raise SystemExit
        _validate_environment()


def test_github_token_validation_empty():
    """Test that empty GitHub token causes exit(1)."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
        with pytest.raises(SystemExit) as exc_info:
            _validate_environment()
        assert exc_info.value.code == 1


def test_github_token_validation_whitespace():
    """Test that whitespace-only GitHub token causes exit(1)."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "   "}):
        with pytest.raises(SystemExit) as exc_info:
            _validate_environment()
        assert exc_info.value.code == 1


def test_github_token_validation_invalid():
    """Test that 'INVALID' GitHub token causes exit(1)."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "INVALID"}):
        with pytest.raises(SystemExit) as exc_info:
            _validate_environment()
        assert exc_info.value.code == 1


def test_github_token_validation_not_provided():
    """Test that missing GitHub token passes validation."""
    with patch.dict(os.environ, {}, clear=True):
        # Should not raise SystemExit when token is not provided
        _validate_environment()


def test_log_file_validation_valid():
    """Test that valid log file path passes validation."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with patch.dict(os.environ, {"LOG_FILE": temp_path}):
            # Should not raise SystemExit
            _validate_environment()
    finally:
        os.unlink(temp_path)


def test_log_file_validation_invalid_path():
    """Test that invalid log file path causes exit(1)."""
    # Use a path that definitely doesn't exist and can't be created
    invalid_path = "/nonexistent/directory/that/cannot/be/created/log.txt"

    with patch.dict(os.environ, {"LOG_FILE": invalid_path}):
        with pytest.raises(SystemExit) as exc_info:
            _validate_environment()
        assert exc_info.value.code == 1


def test_log_file_validation_permission_denied():
    """Test that log file path with permission denied causes exit(1)."""
    # Create a temporary directory and make it read-only
    with tempfile.TemporaryDirectory() as temp_dir:
        read_only_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(read_only_dir)
        os.chmod(read_only_dir, 0o444)  # Read-only

        log_file_path = os.path.join(read_only_dir, "log.txt")

        with patch.dict(os.environ, {"LOG_FILE": log_file_path}):
            with pytest.raises(SystemExit) as exc_info:
                _validate_environment()
            assert exc_info.value.code == 1


def test_log_file_validation_not_provided():
    """Test that missing log file passes validation (uses stderr)."""
    with patch.dict(os.environ, {}, clear=True):
        # Should not raise SystemExit when LOG_FILE is not provided
        _validate_environment()


def test_both_validations_together():
    """Test both validations work together."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "valid_token_123",
            "LOG_FILE": temp_path
        }):
            # Should not raise SystemExit
            _validate_environment()
    finally:
        os.unlink(temp_path)


def test_both_validations_fail():
    """Test that both invalid values cause exit(1)."""
    invalid_path = "/nonexistent/directory/log.txt"

    with patch.dict(os.environ, {
        "GITHUB_TOKEN": "INVALID",
        "LOG_FILE": invalid_path
    }):
        with pytest.raises(SystemExit) as exc_info:
            _validate_environment()
        assert exc_info.value.code == 1


def test_cli_command_validation():
    """Test that CLI commands validate environment variables."""
    # Test that install command validates environment
    with patch.dict(os.environ, {"GITHUB_TOKEN": "INVALID"}):
        with pytest.raises(SystemExit) as exc_info:
            app(["install"])
        assert exc_info.value.code == 1

    # Test that test command validates environment
    with patch.dict(os.environ, {"GITHUB_TOKEN": "INVALID"}):
        with pytest.raises(SystemExit) as exc_info:
            app(["test"])
        assert exc_info.value.code == 1


def test_run_script_validation():
    """Test that the run script validates environment variables."""
    # Create a temporary URL file for testing
    with tempfile.NamedTemporaryFile(
        mode='w', delete=False, suffix='.txt'
    ) as temp_file:
        temp_file.write("https://huggingface.co/test/model")
        url_file_path = temp_file.name

    try:
        # Test with invalid GitHub token
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = "INVALID"

        result = subprocess.run([
            sys.executable, "run", url_file_path
        ], env=env, capture_output=True, text=True)

        assert result.returncode == 1

        # Test with invalid log file
        env = os.environ.copy()
        env["LOG_FILE"] = "/nonexistent/path/log.txt"

        result = subprocess.run([
            sys.executable, "run", url_file_path
        ], env=env, capture_output=True, text=True)

        assert result.returncode == 1

    finally:
        os.unlink(url_file_path)
