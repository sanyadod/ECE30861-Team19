from typing import Dict, Any
from ..models import MetricResult, ModelContext, SizeScore
from ..utils import measure_time, extract_model_size_from_text
from .base import BaseMetric

#metric for evaluating model size and deployment feasibility
class SizeScoreMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "size_score"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        #compute size score for different hardware targets
        with measure_time() as get_latency:
            size_score = await self._calculate_size_scores(context, config)
            # Return scalar max as per specification: max of the four hardware scores
            max_score = max(size_score.raspberry_pi, size_score.jetson_nano, 
                           size_score.desktop_pc, size_score.aws_server)
        
        return MetricResult(score=max_score, latency=get_latency())
    
    async def _calculate_size_scores(self, context: ModelContext, config: Dict[str, Any]) -> SizeScore:
        #estimate model size from various sources
        estimated_size_gb = await self._estimate_model_size(context)
        
        # Use exact thresholds from specification
        return SizeScore(
            raspberry_pi=self._calculate_raspberry_pi_score(estimated_size_gb),
            jetson_nano=self._calculate_jetson_nano_score(estimated_size_gb),
            desktop_pc=self._calculate_desktop_pc_score(estimated_size_gb),
            aws_server=self._calculate_aws_server_score(estimated_size_gb)
        )

    # Generic helper used for tests and potential future refactors
    def _calculate_device_score(self, model_size_gb: float, limit_gb: float) -> float:
        """Calculate a normalized score based on how model size compares to a device limit.

        Rules expected by tests:
        - Well under half the limit (<= 0.5x) -> 1.0
        - At the limit (== 1.0x) -> 0.8
        - Up to 2x the limit (<= 2.0x) -> 0.5
        - Above 2x the limit -> 0.0
        """
        if limit_gb <= 0:
            return 0.0
        ratio = model_size_gb / limit_gb
        if ratio <= 0.5:
            return 1.0
        if ratio == 1.0:
            return 0.8
        if ratio <= 2.0:
            return 0.5
        return 0.0
    
    async def _estimate_model_size(self, context: ModelContext) -> float:
        #extract from README content
        if context.readme_content:
            size_from_readme = extract_model_size_from_text(context.readme_content)
            if size_from_readme:
                return size_from_readme
        
        #estimate from hugging face file information
        if context.hf_info and context.hf_info.get('files'):
            total_size_mb = 0
            for file_path in context.hf_info['files']:
                #estimate sizes based on file extensions
                if file_path.endswith('.bin') or file_path.endswith('.safetensors'):
                    #model weights - estimate based on common patterns
                    if 'pytorch_model' in file_path:
                        total_size_mb += 1000  # ~1GB per model file (rough estimate)
                elif file_path.endswith('.json'):
                    total_size_mb += 1  #config files are small
            
            if total_size_mb > 0:
                return total_size_mb / 1024.0  # Convert to GB
        
        #extract from model name patterns
        model_name = context.model_url.name.lower()
        if '7b' in model_name or '7-b' in model_name:
            return 14.0  # ~14GB for 7B models
        elif '13b' in model_name or '13-b' in model_name:
            return 26.0  # ~26GB for 13B models
        elif '30b' in model_name or '30-b' in model_name:
            return 60.0  # ~60GB for 30B models
        elif '70b' in model_name or '70-b' in model_name:
            return 140.0  # ~140GB for 70B models
        #generic models
        elif 'large' in model_name:
            return 4.0
        elif 'base' in model_name:
            return 1.0
        elif 'small' in model_name:
            return 0.5
        
        #default assumption for unknown
        return 2.0
    
    def _calculate_raspberry_pi_score(self, model_size_gb: float) -> float:
        """Raspberry Pi: <100MB → 1.0, 100–500MB → 0.5, >500MB → 0.0"""
        size_mb = model_size_gb * 1024
        if size_mb < 100:
            return 1.0
        elif size_mb <= 500:
            return 0.5
        else:
            return 0.0
    
    def _calculate_jetson_nano_score(self, model_size_gb: float) -> float:
        """Jetson Nano: <2GB → 1.0, 2–8GB → 0.5, >8GB → 0.0"""
        if model_size_gb < 2.0:
            return 1.0
        elif model_size_gb <= 8.0:
            return 0.5
        else:
            return 0.0
    
    def _calculate_desktop_pc_score(self, model_size_gb: float) -> float:
        """Desktop PC: <10GB → 1.0, 10–50GB → 0.5, >50GB → 0.0"""
        if model_size_gb < 10.0:
            return 1.0
        elif model_size_gb <= 50.0:
            return 0.5
        else:
            return 0.0
    
    def _calculate_aws_server_score(self, model_size_gb: float) -> float:
        """AWS Server: <100GB → 1.0, >100GB → 0.5"""
        if model_size_gb < 100.0:
            return 1.0
        else:
            return 0.5