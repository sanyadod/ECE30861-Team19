from typing import Dict, Any
from ..models import MetricResult, ModelContext
from ..utils import measure_time, check_readme_sections
from .base import BaseMetric


class RampUpTimeMetric(BaseMetric):
    """Metric for evaluating ease of getting started with the model."""
    
    @property
    def name(self) -> str:
        return "ramp_up_time"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        """Compute ramp-up time score based on documentation quality."""
        with measure_time() as get_latency:
            score = await self._calculate_ramp_up_score(context, config)
        
        return MetricResult(score=score, latency=get_latency())
    
    async def _calculate_ramp_up_score(self, context: ModelContext, config: Dict[str, Any]) -> float:
        """Calculate ramp-up score based on available documentation."""
        if not context.readme_content:
            return 0.1  # Very low score for missing README
        
        readme = context.readme_content
        thresholds = config.get('thresholds', {}).get('ramp_up', {})
        required_sections = thresholds.get('readme_sections', ['usage', 'quickstart', 'examples'])
        example_bonus = thresholds.get('example_code_bonus', 0.2)
        
        # Check for required sections
        section_scores = check_readme_sections(readme, required_sections)
        sections_present = sum(section_scores.values())
        section_score = sections_present / len(required_sections)
        
        # Check for code examples (basic pattern matching)
        readme_lower = readme.lower()
        has_code_examples = any([
            '```python' in readme,
            '```py' in readme,
            'example:' in readme_lower,
            'import ' in readme_lower,
            'from ' in readme_lower,
        ])
        
        # Base score from sections
        base_score = section_score * 0.7
        
        # Bonus for code examples
        if has_code_examples:
            base_score += example_bonus
        
        # Length bonus (more comprehensive docs)
        if len(readme) > 1000:
            base_score += 0.1
        
        return min(1.0, base_score)