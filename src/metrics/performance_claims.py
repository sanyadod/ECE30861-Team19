from typing import Dict, Any
from ..models import MetricResult, ModelContext
from ..utils import measure_time, extract_performance_claims
from .base import BaseMetric


class PerformanceClaimsMetric(BaseMetric):
    """Metric for evaluating documented performance claims and benchmarks."""
    
    @property
    def name(self) -> str:
        return "performance_claims"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        """Compute performance claims score."""
        with measure_time() as get_latency:
            score = await self._calculate_performance_score(context, config)
        
        return MetricResult(score=score, latency=get_latency())
    
    async def _calculate_performance_score(self, context: ModelContext, config: Dict[str, Any]) -> float:
        """Calculate performance claims score based on README analysis."""
        if not context.readme_content:
            return 0.0
        
        thresholds = config.get('thresholds', {}).get('performance', {})
        benchmark_keywords = thresholds.get('benchmark_keywords', ['glue', 'mmlu', 'hellaswag'])
        citation_bonus = thresholds.get('citation_bonus', 0.2)
        numeric_bonus = thresholds.get('numeric_results_bonus', 0.3)
        
        # Extract performance information
        perf_info = extract_performance_claims(context.readme_content, benchmark_keywords)
        
        base_score = 0.0
        
        # Score for benchmark mentions
        benchmarks_found = perf_info['benchmarks_mentioned']
        if benchmarks_found:
            benchmark_score = min(0.5, len(benchmarks_found) * 0.1)
            base_score += benchmark_score
        
        # Bonus for numeric results
        if perf_info['numeric_results']:
            base_score += numeric_bonus
        
        # Bonus for citations/links
        if perf_info['citations']:
            base_score += citation_bonus
        
        # Bonus if model card has performance data
        if context.hf_info and context.hf_info.get('model_index'):
            base_score += 0.2
        
        return min(1.0, base_score)