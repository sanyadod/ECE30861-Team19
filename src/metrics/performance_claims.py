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
        """Calculate performance claims: Reproducible steps → 1.0, Vague claims → 0.5, None → 0.0."""
        if not context.readme_content:
            return 0.0  # None → 0.0
        
        readme_lower = context.readme_content.lower()
        
        # Check for reproducible benchmarks/evals (tables or linked papers with steps/scripts)
        reproducible_indicators = [
            'reproduce', 'reproducible', 'evaluation script', 'eval script',
            'benchmark script', 'benchmark.py', 'eval.py', 'evaluation.py',
            'steps to reproduce', 'reproduction', 'script', 'code',
            'github.com/', 'colab', 'notebook'
        ]
        
        # Check for benchmark tables or structured results
        table_indicators = [
            '|', 'table', 'results', 'benchmark', 'performance',
            'accuracy', 'f1', 'bleu', 'rouge', 'glue', 'squad'
        ]
        
        # Check for papers with methodology
        paper_indicators = [
            'paper', 'arxiv', 'methodology', 'experiment', 'evaluation protocol'
        ]
        
        has_reproducible_steps = any(indicator in readme_lower for indicator in reproducible_indicators)
        has_benchmark_tables = any(indicator in readme_lower for indicator in table_indicators)
        has_paper_references = any(indicator in readme_lower for indicator in paper_indicators)
        
        # Reproducible steps present → 1.0
        if has_reproducible_steps and (has_benchmark_tables or has_paper_references):
            return 1.0
        
        # Only vague claims → 0.5
        vague_claims = [
            'perform', 'achieve', 'outperform', 'better', 'improved',
            'state-of-the-art', 'sota', 'good results', 'competitive'
        ]
        has_vague_claims = (has_benchmark_tables or 
                           any(claim in readme_lower for claim in vague_claims))
        
        if has_vague_claims:
            return 0.5
        
        # None → 0.0
        return 0.0