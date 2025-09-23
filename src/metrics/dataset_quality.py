"""
Dataset quality metric - evaluates quality of referenced datasets.
"""
from typing import Dict, Any, List
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from ..hf_api import HuggingFaceAPI
from .base import BaseMetric


class DatasetQualityMetric(BaseMetric):
    """Metric for evaluating quality of linked datasets."""
    
    @property
    def name(self) -> str:
        return "dataset_quality"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        """Compute dataset quality score."""
        with measure_time() as get_latency:
            score = await self._calculate_dataset_quality_score(context, config)
        
        return MetricResult(score=score, latency=get_latency())
    
    async def _calculate_dataset_quality_score(self, context: ModelContext, config: Dict[str, Any]) -> float:
        """Calculate dataset quality based on linked datasets."""
        if not context.datasets:
            return 0.3  # Default score when no datasets are linked
        
        thresholds = config.get('thresholds', {}).get('dataset_quality_checklist', [])
        quality_checklist = thresholds or [
            'description', 'license', 'splits', 'size', 'known_issues', 'ethics'
        ]
        
        total_score = 0.0
        datasets_analyzed = 0
        
        hf_api = HuggingFaceAPI()
        
        for dataset_url in context.datasets:
            if dataset_url.platform == "huggingface":
                dataset_score = await self._analyze_hf_dataset(dataset_url, hf_api, quality_checklist)
                total_score += dataset_score
                datasets_analyzed += 1
        
        if datasets_analyzed == 0:
            return 0.5  # Medium score for non-HF datasets (can't analyze)
        
        return total_score / datasets_analyzed
    
    async def _analyze_hf_dataset(self, dataset_url, hf_api: HuggingFaceAPI, checklist: List[str]) -> float:
        """Analyze a Hugging Face dataset for quality indicators."""
        # Get dataset README
        readme_content = await hf_api.get_readme_content(dataset_url)
        if not readme_content:
            return 0.2  # Low score for missing README
        
        # Get dataset info from API
        dataset_info = await hf_api.get_dataset_info(dataset_url)
        
        score = 0.0
        readme_lower = readme_content.lower()
        
        # Check each quality criterion
        for criterion in checklist:
            if criterion == 'description':
                # Look for dataset description
                if ('description' in readme_lower or 'overview' in readme_lower or 
                    len(readme_content) > 500):
                    score += 1.0 / len(checklist)
            
            elif criterion == 'license':
                # Check for license information
                if ('license' in readme_lower or 
                    (dataset_info and dataset_info.get('tags') and 
                     any('license:' in tag for tag in dataset_info['tags']))):
                    score += 1.0 / len(checklist)
            
            elif criterion == 'splits':
                # Look for train/test/validation split information
                if any(split in readme_lower for split in ['train', 'test', 'validation', 'split']):
                    score += 1.0 / len(checklist)
            
            elif criterion == 'size':
                # Look for size information
                if any(size_term in readme_lower for size_term in 
                      ['size', 'examples', 'instances', 'samples', 'mb', 'gb']):
                    score += 1.0 / len(checklist)
            
            elif criterion == 'known_issues':
                # Look for known issues or limitations section
                if any(issue_term in readme_lower for issue_term in 
                      ['limitations', 'issues', 'bias', 'concerns', 'warnings']):
                    score += 1.0 / len(checklist)
            
            elif criterion == 'ethics':
                # Look for ethics/bias discussion
                if any(ethics_term in readme_lower for ethics_term in 
                      ['ethics', 'bias', 'fairness', 'responsible', 'considerations']):
                    score += 1.0 / len(checklist)
        
        return score
