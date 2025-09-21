from __future__ import annotations

import re
from typing import Dict, Any, Optional
from ..models import ModelContext, MetricResult
from ..logging_utils import get_logger
from ..utils import measure_time

logger = get_logger()

SECTION_PATTERNS = {
    "eval": re.compile(r"\b(eval(uation)? results?|metrics?)\b", re.I),
    "dataset": re.compile(r"\b(dataset|training data)\b", re.I),
    "method": re.compile(r"\b(method|approach|architecture)\b", re.I),
    "limitations": re.compile(r"\b(limitation|bias|risk)\b", re.I),
    "license": re.compile(r"\blicen[cs]e\b", re.I),
}

METRIC_PAT = re.compile(
    r"\b(accuracy|f1|precision|recall|bleu|rouge|exact match|perplexity)\b",
    re.I,
)


def _text_presence_score(text: str) -> float:
    """Score based on presence of key sections and metric keywords."""
    if not text:
        return 0.0

    present = 0
    total = len(SECTION_PATTERNS)
    for pat in SECTION_PATTERNS.values():
        if pat.search(text):
            present += 1

    # Presence of numeric-like metric values improves confidence
    metrics_present = 1 if METRIC_PAT.search(text) else 0

    # Sections weight 0.8, metric keyword presence 0.2
    score = (present / max(1, total)) * 0.8 + metrics_present * 0.2
    return max(0.0, min(1.0, score))


def _model_index_score(model_index: Optional[Any]) -> float:
    if not model_index:
        return 0.0
    # Any structured evaluation entry indicates better claims quality.
    return 1.0


class PerformanceClaimsMetric:
    """Scores how well performance claims are documented and structured."""

    name = "performance_claims"

    async def compute(self, context: ModelContext) -> MetricResult:
        with measure_time() as get_latency:
            # Prefer cached HF info and README/content placed in context
            hf_info: Dict[str, Any] = context.hf_info or {}
            readme: str = context.readme_content or ""

            text_score = _text_presence_score(readme)
            index_score = _model_index_score(hf_info.get("model_index"))

            # Weighted combination:
            # - README/model card sections and metrics: 0.7
            # - Structured model_index presence: 0.3
            score = 0.7 * text_score + 0.3 * index_score

        return MetricResult(score=max(0.0, min(1.0, score)), latency=get_latency())
