from __future__ import annotations

from typing import Dict, Any
from ..models import ModelContext, MetricResult
from ..logging_utils import get_logger
from ..utils import measure_time

logger = get_logger()


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x = (v - lo) / (hi - lo)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


async def _score_from_repo_analysis(analysis: Dict[str, Any]) -> float:
    # Defaults if analysis sections are missing
    file_analysis = analysis.get("file_analysis", {}) or {}
    structure = analysis.get("structure_analysis", {}) or {}
    docs = analysis.get("documentation_analysis", {}) or {}

    python_files = int(file_analysis.get("python_files", 0))
    test_files = int(file_analysis.get("test_files", 0))
    test_ratio = float(file_analysis.get("test_coverage_estimate", 0.0))
    total_files = int(file_analysis.get("total_files", 0))

    structure_score = float(structure.get("structure_score", 0.0))
    documentation_score = float(docs.get("documentation_score", 0.0))
    readme_length = int(docs.get("readme_length", 0))

    # Heuristics:
    # - structure_score (0..1) weight 0.35
    # - documentation_score (0..1) weight 0.25
    # - test_ratio (0..1) weight 0.25
    # - readme_length normalized by cap 2,000 chars weight 0.15
    readme_norm = _normalize(readme_length, 0, 2000)

    weights = {
        "structure": 0.35,
        "docs": 0.25,
        "tests": 0.25,
        "readme": 0.15,
    }

    score = (
        structure_score * weights["structure"]
        + documentation_score * weights["docs"]
        + test_ratio * weights["tests"]
        + readme_norm * weights["readme"]
    )

    return max(0.0, min(1.0, score))


class CodeQualityMetric:

    name = "code_quality"

    async def compute(self, context: ModelContext) -> MetricResult:
        # If no code repos are associated, return 0 quickly
        if not context.code_repos:
            with measure_time() as get_latency:
                score = 0.0
            return MetricResult(score=score, latency=get_latency())

        # Expect another layer to have attached analysis; but if not, score 0
        # The Git analysis is performed elsewhere to avoid network in metrics.
        # Here, look for an attached attribute placed by the scorer like:
        # context.extra["code_repo_analysis"][repo_name] = analysis_dict
        analyses = getattr(context, "extra", {}).get("code_repo_analysis", {})
        if not analyses:
            with measure_time() as get_latency:
                score = 0.0
            return MetricResult(score=score, latency=get_latency())

        # Aggregate multiple repos: take the best quality as representative,
        # or average—use average to be conservative.
        with measure_time() as get_latency:
            repo_scores = []
            for repo_key, analysis in analyses.items():
                try:
                    s = await _score_from_repo_analysis(analysis)
                    repo_scores.append(s)
                except Exception as e:
                    logger.debug(f"Error scoring repo {repo_key}: {e}")
                    repo_scores.append(0.0)

            score = sum(repo_scores) / len(repo_scores) if repo_scores else 0.0

        return MetricResult(score=max(0.0, min(1.0, score)), latency=get_latency())
