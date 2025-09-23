"""
License score metric - evaluates license compatibility and clarity.
"""
from typing import Dict, Any
from ..models import MetricResult, ModelContext
from ..utils import measure_time, parse_license_from_readme
from .base import BaseMetric


class LicenseScoreMetric(BaseMetric):
    """Metric for evaluating license compatibility and clarity."""
    
    @property
    def name(self) -> str:
        return "license"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        """Compute license score."""
        with measure_time() as get_latency:
            score = await self._calculate_license_score(context, config)
        
        return MetricResult(score=score, latency=get_latency())
    
    async def _calculate_license_score(self, context: ModelContext, config: Dict[str, Any]) -> float:
        """Calculate license score based on compatibility and clarity."""
        thresholds = config.get('thresholds', {}).get('license', {})
        compatible_licenses = thresholds.get('compatible_licenses', ['apache-2.0', 'mit', 'bsd-3-clause'])
        restrictive_penalty = thresholds.get('restrictive_penalty', 0.3)
        missing_penalty = thresholds.get('missing_penalty', 0.7)
        
        # Try to get license from HF info first
        license_info = None
        if context.hf_info and context.hf_info.get('tags'):
            # Look for license in tags
            for tag in context.hf_info['tags']:
                if tag.startswith('license:'):
                    license_info = tag.replace('license:', '')
                    break
        
        # If no HF license, try README
        if not license_info and context.readme_content:
            license_info = parse_license_from_readme(context.readme_content)
        
        if not license_info:
            return 1.0 - missing_penalty  # Penalty for missing license
        
        license_lower = license_info.lower()
        
        # Check for compatible licenses
        for compatible in compatible_licenses:
            if compatible in license_lower:
                return 1.0  # Full score for compatible license
        
        # Check for restrictive licenses
        restrictive_terms = ['gpl', 'agpl', 'commercial', 'proprietary', 'all rights reserved']
        for term in restrictive_terms:
            if term in license_lower:
                return 1.0 - restrictive_penalty
        
        # Unknown license - medium score
        return 0.5