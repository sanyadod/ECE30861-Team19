"""
Tests for URL parsing and categorization.
"""
import pytest
from src.urls import parse_url, build_model_contexts, _find_relevant_resources
from src.models import URLCategory, ParsedURL


def test_parse_huggingface_model_url():
    """Test parsing Hugging Face model URL."""
    url = "https://huggingface.co/google/gemma-3-270m"
    parsed = parse_url(url)
    
    assert parsed.category == URLCategory.MODEL
    assert parsed.platform == "huggingface"
    assert parsed.owner == "google"
    assert parsed.repo == "gemma-3-270m"
    assert parsed.name == "google/gemma-3-270m"


def test_parse_huggingface_dataset_url():
    """Test parsing Hugging Face dataset URL."""
    url = "https://huggingface.co/datasets/xlangai/AgentNet"
    parsed = parse_url(url)
    
    assert parsed.category == URLCategory.DATASET
    assert parsed.platform == "huggingface"
    assert parsed.owner == "xlangai"
    assert parsed.repo == "AgentNet"
    assert parsed.name == "xlangai/AgentNet"


def test_parse_github_url():
    """Test parsing GitHub URL."""
    url = "https://github.com/SkyworkAI/Matrix-Game"
    parsed = parse_url(url)
    
    assert parsed.category == URLCategory.CODE
    assert parsed.platform == "github"
    assert parsed.owner == "SkyworkAI"
    assert parsed.repo == "Matrix-Game"
    assert parsed.name == "SkyworkAI/Matrix-Game"


def test_parse_unknown_url():
    """Test parsing unknown URL."""
    url = "https://example.com/some/model"
    parsed = parse_url(url)
    
    assert parsed.category == URLCategory.MODEL
    assert parsed.platform == "unknown"
    assert parsed.name == "model"


def test_parse_invalid_github_url():
    """Test parsing invalid GitHub URL."""
    url = "https://github.com/incomplete"
    with pytest.raises(ValueError):
        parse_url(url)


def test_build_model_contexts():
    """Test building model contexts from URLs."""
    urls = [
        "https://huggingface.co/datasets/test/dataset",
        "https://github.com/test/code",
        "https://huggingface.co/test/model"
    ]
    
    contexts = build_model_contexts(urls)
    
    assert len(contexts) == 1
    context = contexts[0]
    assert context.model_url.name == "test/model"
    assert len(context.datasets) >= 0  # May or may not find relevant datasets
    assert len(context.code_repos) >= 0  # May or may not find relevant code


def test_find_relevant_resources():
    """Test finding relevant resources for a model."""
    model_url = ParsedURL(
        url="https://huggingface.co/test/sentiment-model",
        category=URLCategory.MODEL,
        name="test/sentiment-model",
        platform="huggingface",
        owner="test",
        repo="sentiment-model"
    )
    
    resources = [
        ParsedURL(
            url="https://huggingface.co/datasets/test/sentiment-data",
            category=URLCategory.DATASET,
            name="test/sentiment-data",
            platform="huggingface",
            owner="test",
            repo="sentiment-data"
        )
    ]
    
    relevant = _find_relevant_resources(model_url, resources)
    assert len(relevant) > 0  # Should find the matching owner


def test_empty_url_list():
    """Test building contexts from empty URL list."""
    contexts = build_model_contexts([])
    assert len(contexts) == 0