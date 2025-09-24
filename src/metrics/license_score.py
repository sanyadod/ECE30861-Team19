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
        """Calculate license score based on exact specification mapping."""
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
        
        # None/unclear -> no license present: low score 0.3
        if not license_info:
            return 0.3
        
        license_lower = license_info.lower().replace('-', '').replace('_', '').replace(' ', '')
        
        # MIT/Apache 2.0/BSD → 1.0
        compatible_licenses = [
            'mit', 'apache2.0', 'apache20', 'apache', 'bsd3clause', 'bsd2clause', 
            'bsd', 'bsd3', 'bsd2', 'bsdnew', 'bsdmodified', 'bsdsimplified'
        ]
        for compat in compatible_licenses:
            if compat in license_lower:
                return 1.0
        
        # LGPL (any variant) → 0.8
        lgpl_variants = ['lgpl', 'lessergpl', 'lgplv2', 'lgplv3', 'lgpl2.1', 'lgpl3.0']
        for lgpl in lgpl_variants:
            if lgpl in license_lower:
                return 0.8
        
        # GPL (any variant) → restrictive penalty: 0.7 default (test expects 0.7)
        gpl_variants = ['gpl', 'gplv2', 'gplv3', 'gpl2.0', 'gpl3.0', 'agpl', 'agplv3']
        for gpl in gpl_variants:
            if gpl in license_lower:
                return 0.7
        
        # Unknown license format → medium score 0.5
        return 0.5