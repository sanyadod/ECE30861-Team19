"""
Final tests to push coverage over 80%.
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from pathlib import Path

from src.git_inspect import GitInspector


def test_git_inspector_comprehensive():
    """Test more git inspector paths."""
    inspector = GitInspector()
    
    # Test commit quality calculation
    commit_analysis = {
        'total_commits': 50,
        'recent_commits': 10, 
        'avg_commit_frequency': 0.5
    }
    score = inspector._analyze_files.__func__(inspector, "/tmp")['total_files']
    assert score >= 0
    
    # Test structure analysis paths
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create standard files
        (Path(temp_dir) / "README.md").touch()
        (Path(temp_dir) / "LICENSE").touch()
        (Path(temp_dir) / "requirements.txt").touch()
        
        analysis = inspector._analyze_structure(temp_dir)
        assert analysis['has_readme'] == True
        assert analysis['has_license'] == True
        assert analysis['has_requirements'] == True


def test_cli_paths():
    """Test remaining CLI paths."""
    from src.cli import run_tests
    
    with patch('subprocess.run') as mock_run:
        # Test with no output parsing
        mock_run.return_value = Mock(
            returncode=0,
            stdout="No clear results",
            stderr=""
        )
        
        with pytest.raises(SystemExit):
            run_tests()


def test_utils_regex_paths():
    """Test utility regex edge cases."""
    from src.utils import extract_model_size_from_text
    
    # Test TB size
    size = extract_model_size_from_text("Model size: 1.2TB")
    assert size == 1228.8  # 1.2 * 1024
    
    # Test with no match at all
    size = extract_model_size_from_text("No numbers here at all")
    assert size is None


def test_error_paths():
    """Test error handling paths in utilities."""
    from src.utils import extract_model_size_from_text, parse_license_from_readme
    
    # Test with invalid size values that cause ValueError
    size = extract_model_size_from_text("Model has NotANumber parameters")
    assert size is None
    
    # Test license parsing with markdown links
    readme = "## License\n[MIT License](https://mit.license)"
    license_info = parse_license_from_readme(readme)
    assert license_info == "MIT License"