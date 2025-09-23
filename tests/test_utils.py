"""
Tests for utility functions.
"""
import pytest
from src.utils import (
    extract_model_size_from_text, 
    parse_license_from_readme,
    check_readme_sections,
    extract_performance_claims,
    measure_time
)


def test_extract_model_size_7b():
    """Test extracting 7B model size."""
    text = "This is a 7B parameter model"
    size = extract_model_size_from_text(text)
    assert size == 14.0  # 7 * 2GB


def test_extract_model_size_million():
    """Test extracting million parameter model size."""
    text = "This model has 270 million parameters"
    size = extract_model_size_from_text(text)
    assert size == 0.54  # 270 * 0.002GB


def test_extract_model_size_gb():
    """Test extracting size in GB."""
    text = "Model size is 13.5GB"
    size = extract_model_size_from_text(text)
    assert size == 13.5


def test_extract_model_size_none():
    """Test extracting size from text with no size info."""
    text = "This is a model without size information"
    size = extract_model_size_from_text(text)
    assert size is None


def test_parse_license_from_readme():
    """Test parsing license from README."""
    readme = """
    # Model
    
    ## License
    This model is licensed under MIT License.
    """
    
    license_info = parse_license_from_readme(readme)
    assert license_info is not None
    assert "MIT License" in license_info


def test_parse_license_from_readme_none():
    """Test parsing license from README without license section."""
    readme = "# Model\n\nThis is a model."
    
    license_info = parse_license_from_readme(readme)
    assert license_info is None


def test_check_readme_sections():
    """Test checking README sections."""
    readme = """
    # Model
    
    ## Usage
    How to use...
    
    ## Examples  
    Example code...
    """
    
    sections = ['usage', 'examples', 'installation']
    result = check_readme_sections(readme, sections)
    
    assert result['usage'] == True
    assert result['examples'] == True
    assert result['installation'] == False


def test_extract_performance_claims():
    """Test extracting performance claims."""
    readme = """
    # Model
    
    Performance on GLUE benchmark: 82.3%
    MMLU score: 0.85
    
    See paper: https://arxiv.org/abs/2301.00001
    """
    
    keywords = ['glue', 'mmlu']
    claims = extract_performance_claims(readme, keywords)
    
    assert 'glue' in claims['benchmarks_mentioned']
    assert 'mmlu' in claims['benchmarks_mentioned']
    assert claims['numeric_results'] == True
    assert claims['citations'] == True


def test_measure_time_context_manager():
    """Test measure_time context manager."""
    with measure_time() as get_latency:
        # Simulate some work
        import time
        time.sleep(0.01)  # 10ms
    
    latency = get_latency()
    assert latency >= 10  # Should be at least 10ms
    assert isinstance(latency, int)


def test_extract_performance_claims_empty():
    """Test extracting performance claims from empty text."""
    claims = extract_performance_claims("", ['glue'])
    
    assert claims['benchmarks_mentioned'] == []
    assert claims['numeric_results'] == False
    assert claims['citations'] == False