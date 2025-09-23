"""
Dataset and code score metric - evaluates linked training data and example code.
"""
from typing import Dict, Any
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric


class DatasetAndCodeScoreMetric(BaseMetric):
    """Metric for evaluating linked datasets and code quality."""
    
    @property
    def name(self) -> str:
        return "dataset_and_code_score"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        """Compute dataset and code linkage score."""
        with measure_time() as get_latency:
            score = await self._calculate_dataset_code_score(context, config)
        
        return MetricResult(score=score, latency=get_latency())
    
    async def _calculate_dataset_code_score(self, context: ModelContext, config: Dict[str, Any]) -> float:
        """Calculate score based on linked datasets and code repositories."""
        thresholds = config.get('thresholds', {}).get('dataset_code', {})
        dataset_bonus = thresholds.get('dataset_link_bonus', 0.4)
        code_bonus = thresholds.get('code_example_bonus', 0.3)
        both_bonus = thresholds.get('both_present_bonus', 0.3)
        
        score = 0.1  # Base score for having a model
        
        # Score for linked datasets
        has_datasets = bool(context.datasets)
        if has_datasets:
            score += dataset_bonus
            
            # Additional bonus for multiple datasets
            if len(context.datasets) > 1:
                score += 0.1
        
        # Score for linked code repositories
        has_code = bool(context.code_repos)
        if has_code:
            score += code_bonus
            
            # Additional bonus for multiple code repos
            if len(context.code_repos) > 1:
                score += 0.1
        
        # Bonus for having both datasets and code
        if has_datasets and has_code:
            score += both_bonus
        
        # Check README for explicit dataset/code mentions
        if context.readme_content:
            readme_lower = context.readme_content.lower()
            
            dataset_mentions = any(term in readme_lower for term in [
                'training data', 'dataset', 'benchmark', 'corpus', 'training set'
            ])
            
            code_mentions = any(term in readme_lower for term in [
                'training code', 'fine-tuning', 'example', 'tutorial', 'notebook'
            ])
            
            if dataset_mentions:
                score += 0.1
            if code_mentions:
                score += 0.1
        
        return min(1.0, score)
