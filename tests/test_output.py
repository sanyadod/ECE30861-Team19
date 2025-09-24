"""
Tests for NDJSON output functionality.
"""

import json
from io import StringIO
from unittest.mock import patch

import pytest

from src.models import AuditResult, SizeScore
from src.output import NDJSONOutputter


@pytest.fixture
def outputter():
    """Create NDJSONOutputter instance."""
    return NDJSONOutputter()


@pytest.fixture
def sample_result():
    """Create sample audit result."""
    size_score = SizeScore(
        raspberry_pi=0.5, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0
    )

    return AuditResult(
        name="test/model",
        net_score=0.75,
        net_score_latency=500,
        ramp_up_time=0.8,
        ramp_up_time_latency=100,
        bus_factor=0.6,
        bus_factor_latency=200,
        performance_claims=0.7,
        performance_claims_latency=150,
        license=1.0,
        license_latency=50,
        size_score=size_score,
        size_score_latency=300,
        dataset_and_code_score=0.8,
        dataset_and_code_score_latency=250,
        dataset_quality=0.7,
        dataset_quality_latency=180,
        code_quality=0.6,
        code_quality_latency=220,
    )


def test_output_single_result(outputter, sample_result):
    """Test outputting single result to stdout."""
    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        outputter.output_single_result(sample_result)

    output = captured_output.getvalue().strip()

    # Parse JSON to verify it's valid
    parsed = json.loads(output)
    assert parsed["name"] == "test/model"
    assert parsed["category"] == "MODEL"
    assert parsed["net_score"] == 0.75


def test_output_multiple_results(outputter, sample_result):
    """Test outputting multiple results to stdout."""
    results = [sample_result, sample_result]

    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        outputter.output_results(results)

    output_lines = captured_output.getvalue().strip().split("\n")
    assert len(output_lines) == 2

    # Verify each line is valid JSON
    for line in output_lines:
        parsed = json.loads(line)
        assert parsed["name"] == "test/model"


def test_json_schema_compliance(outputter, sample_result):
    """Test that output complies with required JSON schema."""
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        outputter.output_single_result(sample_result)

    output = captured_output.getvalue().strip()
    parsed = json.loads(output)

    # Check all required fields are present
    required_fields = [
        "name",
        "category",
        "net_score",
        "net_score_latency",
        "ramp_up_time",
        "ramp_up_time_latency",
        "bus_factor",
        "bus_factor_latency",
        "performance_claims",
        "performance_claims_latency",
        "license",
        "license_latency",
        "size_score",
        "size_score_latency",
        "dataset_and_code_score",
        "dataset_and_code_score_latency",
        "dataset_quality",
        "dataset_quality_latency",
        "code_quality",
        "code_quality_latency",
    ]

    for field in required_fields:
        assert field in parsed, f"Missing required field: {field}"

    # Check size_score structure
    size_score = parsed["size_score"]
    assert "raspberry_pi" in size_score
    assert "jetson_nano" in size_score
    assert "desktop_pc" in size_score
    assert "aws_server" in size_score

    # Check category is always "MODEL"
    assert parsed["category"] == "MODEL"
