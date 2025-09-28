from typing import Any, Dict

from ..models import MetricResult, ModelContext, SizeScore
from ..utils import extract_model_size_from_text, measure_time
from .base import BaseMetric
import math


class SizeScoreMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "size_score"

    async def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute size score for different hardware targets."""
        with measure_time() as get_latency:
            size_score = await self._calculate_size_scores(context, config)
        
        # Return MetricResult with SizeScore object as the score
        return MetricResult(
            score=size_score,
            latency=get_latency()
        )

    async def _calculate_size_scores(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> SizeScore:
        """Estimate model size from various sources."""
        estimated_size_gb = await self._estimate_model_size(context)

        # Get thresholds from config, with fallbacks
        size_limits = config.get("thresholds", {}).get("size_limits", {})
        raspberry_pi_limit = size_limits.get("raspberry_pi", 2.0)
        jetson_nano_limit = size_limits.get("jetson_nano", 8.0)
        desktop_pc_limit = size_limits.get("desktop_pc", 32.0)
        aws_server_limit = size_limits.get("aws_server", 128.0)

        return SizeScore(
            raspberry_pi=self._calculate_device_score(estimated_size_gb, raspberry_pi_limit),
            jetson_nano=self._calculate_device_score(estimated_size_gb, jetson_nano_limit),
            desktop_pc=self._calculate_device_score(estimated_size_gb, desktop_pc_limit),
            aws_server=self._calculate_device_score(estimated_size_gb, aws_server_limit),
        )

    def _calculate_device_score(self, model_size_gb: float, limit_gb: float) -> float:
        """Calculate normalized score versus device limit.
        
        Score ranges from 0.0 to 1.0:
        - Models at or under limit: score = 1.0
        - Models 2x over limit: score ≈ 0.5  
        - Models 5x over limit: score ≈ 0.2
        - Models 10x+ over limit: score → 0.0
        """
        if limit_gb <= 0:
            return 0.0
        
        if model_size_gb <= 0:
            return 1.0
        
        ratio = model_size_gb / limit_gb
        
        # Models within or at limit get perfect score
        if ratio <= 1.0:
            return 1.0
        
        # Exponential decay for oversized models
        # This creates a smooth curve: 2x=0.5, 4x=0.25, 8x=0.125, etc.
        score = 1.0 / ratio
        return round(max(score, 0.0), 2)

    async def _estimate_model_size(self, context: ModelContext) -> float:
        """Estimate model size from various sources."""
        
        # First, try to use the utility function if available
        try:
            size_from_text = extract_model_size_from_text(context.model_url.name)
            if size_from_text and size_from_text > 0:
                return size_from_text
        except Exception:
            pass  # Fall back to other methods
        
        # Try HuggingFace file information with better size estimation
        if context.hf_info and context.hf_info.get("files"):
            total_size_gb = 0.0
            model_files = 0
            
            for file_path in context.hf_info["files"]:
                if file_path.endswith((".bin", ".safetensors")):
                    # Estimate based on file size if available, otherwise use heuristics
                    file_info = context.hf_info.get("file_info", {}).get(file_path, {})
                    if "size" in file_info:
                        # Convert bytes to GB
                        total_size_gb += file_info["size"] / (1024**3)
                    else:
                        # Fallback: estimate ~1-2GB per model file
                        total_size_gb += 1.5
                    model_files += 1
                elif file_path.endswith(".h5"):
                    total_size_gb += 1.0  # H5 files tend to be smaller
                    model_files += 1
                elif file_path.endswith((".json", ".txt", ".md")):
                    total_size_gb += 0.001  # Config files are tiny
                    
            if model_files > 0:
                return max(total_size_gb, 0.1)  # Minimum realistic size
        
        # Extract from model name patterns with improved regex
        model_name = context.model_url.name.lower()
        
        import re
        
        # Try billion parameter patterns first (more common for larger models)
        b_patterns = [
            r'(\d+(?:\.\d+)?)b(?![a-z])',  # Match 7b, 13b, 1.3b but not "mobile"
            r'(\d+(?:\.\d+)?)-b\b',        # Match 7-b, 13-b  
            r'(\d+(?:\.\d+)?)_b\b',        # Match 7_b, 13_b
            r'(\d+(?:\.\d+)?)\s*billion',  # Match "7 billion", "1.3 billion"
        ]
        
        for pattern in b_patterns:
            match = re.search(pattern, model_name)
            if match:
                param_count = float(match.group(1))
                # Use 2GB per billion parameters (standard estimate)
                return param_count * 2.0
        
        # Try million parameter patterns
        m_patterns = [
            r'(\d+(?:\.\d+)?)m(?![a-z])',  # Match 125m, 350m but not "model"
            r'(\d+(?:\.\d+)?)-m\b',        # Match 125-m
            r'(\d+(?:\.\d+)?)_m\b',        # Match 125_m  
            r'(\d+(?:\.\d+)?)\s*million',  # Match "125 million"
        ]
        
        for pattern in m_patterns:
            match = re.search(pattern, model_name)
            if match:
                param_count = float(match.group(1))
                return param_count * 0.002  # 2MB per million parameters
        
        # Look for size indicators in GB/MB
        size_patterns = [
            r'(\d+(?:\.\d+)?)gb',  # Match 3gb, 1.5gb
            r'(\d+(?:\.\d+)?)g\b', # Match 3g
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, model_name)
            if match:
                return float(match.group(1))
        
        mb_patterns = [
            r'(\d+(?:\.\d+)?)mb',  # Match 500mb, 1500mb
        ]
        
        for pattern in mb_patterns:
            match = re.search(pattern, model_name)
            if match:
                return float(match.group(1)) / 1024  # Convert MB to GB
        
        # Model family/size heuristics (improved estimates)
        size_keywords = {
            ('xxl', 'ultra', 'giant'): 12.0,
            ('xl', 'large'): 3.0,
            ('base', 'medium'): 1.2,
            ('small',): 0.5,
            ('mini', 'tiny', 'nano'): 0.2,
        }
        
        for keywords, size in size_keywords.items():
            if any(keyword in model_name for keyword in keywords):
                return size
        
        # Check for common model architectures
        if any(arch in model_name for arch in ['gpt-3', 'gpt3']):
            return 6.0  # GPT-3 style models are typically large
        elif any(arch in model_name for arch in ['bert-large']):
            return 1.3
        elif any(arch in model_name for arch in ['bert-base']):
            return 0.4
        elif 'distil' in model_name:
            return 0.3  # Distilled models are smaller
        
        # Default fallback
        return 1.0