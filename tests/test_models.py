import pytest
from pydantic import ValidationError

from src.models import (
    AuditResult,
    MetricResult,
    ModelContext,
    ParsedURL,
    SizeScore,
    URLCategory,
)


def test_url_category_enum():
    assert URLCategory.MODEL == "MODEL"
    assert URLCategory.DATASET == "DATASET"
    assert URLCategory.CODE == "CODE"


def test_parsed_url_creation():
    url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
        owner="test",
        repo="model",
    )

    assert url.url == "https://huggingface.co/test/model"
    assert url.category == URLCategory.MODEL
    assert url.name == "test/model"
    assert url.platform == "huggingface"
    assert url.owner == "test"
    assert url.repo == "model"


def test_size_score_validation():
    # valid scores
    size_score = SizeScore(
        raspberry_pi=0.5, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0
    )

    assert size_score.raspberry_pi == 0.5
    assert size_score.jetson_nano == 0.7

    # invalid scores out of range
    with pytest.raises(ValidationError):
        SizeScore(
            raspberry_pi=1.5, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0  # > 1.0
        )

    with pytest.raises(ValidationError):
        SizeScore(
            raspberry_pi=-0.1, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0  # < 0.0
        )


def test_metric_result_validation():
    # valid result
    result = MetricResult(score=0.8, latency=150)
    assert result.score == 0.8
    assert result.latency == 150

    # invalid score
    with pytest.raises(ValidationError):
        MetricResult(score=1.1, latency=150)  # score > 1.0

    # invalid latency
    with pytest.raises(ValidationError):
        MetricResult(score=0.8, latency=-10)  # latency < 0


def test_audit_result_creation():
    size_score = SizeScore(
        raspberry_pi=0.5, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0
    )

    result = AuditResult(
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

    assert result.name == "test/model"
    assert result.category == "MODEL"  # default
    assert result.net_score == 0.75
    assert result.size_score == size_score


def test_model_context_creation():
    """Test ModelContext model creation."""
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
    )

    context = ModelContext(model_url=model_url)

    assert context.model_url == model_url
    assert context.datasets == []
    assert context.code_repos == []
    assert context.hf_info is None
    assert context.readme_content is None
    assert context.config_data is None


def test_model_context_with_resources():
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
    )

    dataset_url = ParsedURL(
        url="https://huggingface.co/datasets/test/dataset",
        category=URLCategory.DATASET,
        name="test/dataset",
        platform="huggingface",
    )

    code_url = ParsedURL(
        url="https://github.com/test/code",
        category=URLCategory.CODE,
        name="test/code",
        platform="github",
    )

    context = ModelContext(
        model_url=model_url, datasets=[dataset_url], code_repos=[code_url]
    )

    assert len(context.datasets) == 1
    assert len(context.code_repos) == 1
    assert context.datasets[0] == dataset_url
    assert context.code_repos[0] == code_url
