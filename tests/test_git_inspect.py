"""
Tests for Git repository inspection.
"""
import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch

from src.git_inspect import GitInspector
from src.models import ParsedURL, URLCategory


@pytest.fixture
def git_inspector():
    """Create GitInspector instance for testing."""
    return GitInspector()


@pytest.fixture
def github_url():
    """Create GitHub URL for testing."""
    return ParsedURL(
        url="https://github.com/test/repo",
        category=URLCategory.CODE,
        name="test/repo",
        platform="github",
        owner="test",
        repo="repo"
    )


def test_git_inspector_init(git_inspector):
    """Test GitInspector initialization."""
    assert git_inspector.cache_dir is not None
    assert os.path.exists(git_inspector.cache_dir)


@patch('src.git_inspect.porcelain.clone')
def test_clone_repo_success(mock_clone, git_inspector, github_url):
    """Test successful repository cloning."""
    repo_path = git_inspector.clone_repo(github_url)
    
    assert repo_path is not None
    assert github_url.owner in repo_path
    assert github_url.repo in repo_path
    mock_clone.assert_called_once()


def test_clone_repo_non_github(git_inspector):
    """Test cloning non-GitHub repository."""
    non_github_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface"
    )
    
    result = git_inspector.clone_repo(non_github_url)
    assert result is None


@patch('src.git_inspect.porcelain.clone')
def test_clone_repo_failure(mock_clone, git_inspector, github_url):
    """Test repository cloning failure."""
    mock_clone.side_effect = Exception("Clone failed")
    
    result = git_inspector.clone_repo(github_url)
    assert result is None


def test_analyze_repository_not_git(git_inspector):
    """Test analyzing non-git directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        analysis = git_inspector.analyze_repository(temp_dir)
        
        # Should return empty analysis
        assert analysis['commit_analysis']['total_commits'] == 0
        assert analysis['contributor_analysis']['unique_authors'] == 0


def test_cleanup(git_inspector):
    """Test cache cleanup."""
    cache_dir = git_inspector.cache_dir
    assert os.path.exists(cache_dir)
    
    git_inspector.cleanup()
    assert not os.path.exists(cache_dir)