"""
Targeted tests for CodeQualityMetric branch coverage.
"""

from src.metrics.code_quality import CodeQualityMetric


def test_analyze_code_repository_scoring_mix():
    metric = CodeQualityMetric()

    # Craft an analysis dict that exercises weighting and capping
    analysis = {
        "structure_analysis": {"structure_score": 0.8},  # 25% weight
        "documentation_analysis": {"documentation_score": 0.6},  # 25% weight
        "file_analysis": {
            "python_files": 50,
            "test_files": 30,
            "total_lines_of_code": 20000,
        },
        "commit_analysis": {
            "total_commits": 21,
            "recent_commits": 5,
            "avg_commit_frequency": 0.2,
        },
    }

    # Use private helpers directly to compute expected portions
    file_score = metric._calculate_file_quality_score(
        analysis["file_analysis"], {"min_test_coverage": 0.5}
    )
    commit_score = metric._calculate_commit_quality_score(analysis["commit_analysis"])

    # Combine as in _analyze_code_repository
    combined = min(
        1.0,
        0.25 * analysis["structure_analysis"]["structure_score"]
        + 0.25 * analysis["documentation_analysis"]["documentation_score"]
        + 0.25 * file_score
        + 0.25 * commit_score,
    )

    # Call the target function
    score = metric._analyze_code_repository(
        "/dev/null",
        type("I", (), {"analyze_repository": lambda *_: analysis})(),
        {"min_test_coverage": 0.5},
    )
    assert abs(score - combined) < 1e-6


def test_calculate_file_quality_score_thresholds():
    metric = CodeQualityMetric()

    # High coverage and within file/loc ranges
    fa = {"python_files": 20, "test_files": 10, "total_lines_of_code": 5000}
    s = metric._calculate_file_quality_score(fa, {"min_test_coverage": 0.5})
    # 0.4 (coverage) + 0.3 (files) + 0.3 (loc) = 1.0
    assert abs(s - 1.0) < 1e-6

    # Medium coverage threshold path
    fa = {"python_files": 20, "test_files": 6, "total_lines_of_code": 100}
    s = metric._calculate_file_quality_score(fa, {"min_test_coverage": 0.5})
    # 0.2 (half coverage) + 0.3 (files) + 0.1 (loc low but >0)
    assert abs(s - 0.6) < 1e-6

    # Few files path
    fa = {"python_files": 2, "test_files": 1, "total_lines_of_code": 10}
    s = metric._calculate_file_quality_score(fa, {"min_test_coverage": 0.5})
    # 0.4 (coverage) + 0.1 (files >0) + 0.1 (loc >0)
    assert abs(s - 0.6) < 1e-6


def test_calculate_commit_quality_score_branches():
    metric = CodeQualityMetric()

    # High activity path
    s = metric._calculate_commit_quality_score(
        {"total_commits": 25, "recent_commits": 6, "avg_commit_frequency": 0.15}
    )
    # 0.3 + 0.3 + 0.4 = 1.0
    assert abs(s - 1.0) < 1e-6

    # Lower activity path
    s = metric._calculate_commit_quality_score(
        {"total_commits": 6, "recent_commits": 1, "avg_commit_frequency": 0.05}
    )
    # 0.2 + 0.2 + 0.2 = 0.6
    assert abs(s - 0.6) < 1e-6
