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
        """Calculate ramp-up score based on 4 specific criteria."""
        score = 0.0
        criteria_count = 4
        
        # Criterion 1: README present
        if context.readme_content:
            score += 1.0 / criteria_count
            readme_lower = context.readme_content.lower()
        else:
            # No README present - return low baseline score
            return 0.1
        
        # Criterion 2: Install instructions
        install_indicators = [
            'install', 'pip install', 'conda install', 'npm install', 'yarn install',
            'setup', 'installation', 'getting started', 'requirements', 'dependencies'
        ]
        if any(indicator in readme_lower for indicator in install_indicators):
            score += 1.0 / criteria_count
        
        # Criterion 3: Training/evaluation examples
        training_indicators = [
            'training', 'train', 'fine-tuning', 'fine tuning', 'finetune', 
            'evaluation', 'eval', 'benchmark', 'test', 'validate'
        ]
        if any(indicator in readme_lower for indicator in training_indicators):
            score += 1.0 / criteria_count
        
        # Criterion 4: API usage examples
        api_indicators = [
            'usage', 'example', 'how to use', 'quickstart', 'tutorial',
            'from transformers', 'import', 'model.', 'pipeline',
            '```python', '```py', 'api', 'inference'
        ]
        if any(indicator in readme_lower for indicator in api_indicators):
            score += 1.0 / criteria_count
        
        # Bonus: Check for tutorials/examples area with ≥1 item (+0.1 bonus, cap at 1.0)
        if context.hf_info and context.hf_info.get('files'):
            files = context.hf_info['files']
            has_examples = any(
                'example' in file_path.lower() or 'tutorial' in file_path.lower() or 
                'notebook' in file_path.lower() or file_path.endswith('.ipynb')
                for file_path in files
            )
            if has_examples:
                score += 0.1
        
        return min(1.0, score)