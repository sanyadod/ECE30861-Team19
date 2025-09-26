from typing import Any, Dict

from ..models import MetricResult, ModelContext, SizeScore
from ..utils import extract_model_size_from_text, measure_time
from .base import BaseMetric


# metric for evaluating model size and deployment feasibility
class SizeScoreMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "size_score"

    async def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        # compute size score for different hardware targets
        with measure_time() as get_latency:
            size_score = await self._calculate_size_scores(context, config)
            # Return scalar max as per specification: max of the four hardware scores
            # max_score = max(
            #     size_score.raspberry_pi,
            #     size_score.jetson_nano,
            #     size_score.desktop_pc,
            #     size_score.aws_server,
            # )

        # return MetricResult(
        #     score=max_score, 
        #     latency=get_latency())
        return {
            "size_score": {
                "raspberry_pi": size_score.raspberry_pi,
                "jetson_nano": size_score.jetson_nano,
                "desktop_pc": size_score.desktop_pc,
                "aws_server": size_score.aws_server,
            },
        "size_score_latency": get_latency(),
        }

    async def _calculate_size_scores(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> SizeScore:
        """Calculate size scores with improved handling for different model types and use cases."""
        # estimate model size from various sources
        estimated_size_gb = await self._estimate_model_size(context)

        # Get thresholds from config, with fallbacks to hardcoded values
        size_limits = config.get("thresholds", {}).get("size_limits", {})
        raspberry_pi_limit = size_limits.get("raspberry_pi", 2.0)
        jetson_nano_limit = size_limits.get("jetson_nano", 8.0)
        desktop_pc_limit = size_limits.get("desktop_pc", 32.0)
        aws_server_limit = size_limits.get("aws_server", 128.0)

        # Adjust limits based on model size only (no hard-coded model types)
        # For edge devices, be more lenient with small models
        if estimated_size_gb < 0.5:  # Very small models
            raspberry_pi_limit *= 1.5  # More lenient for tiny models
            jetson_nano_limit *= 1.2

        # Use config thresholds with improved device score calculation
        return SizeScore(
            raspberry_pi=self._calculate_device_score(estimated_size_gb, raspberry_pi_limit),
            jetson_nano=self._calculate_device_score(estimated_size_gb, jetson_nano_limit),
            desktop_pc=self._calculate_device_score(estimated_size_gb, desktop_pc_limit),
            aws_server=self._calculate_device_score(estimated_size_gb, aws_server_limit),
        )

    # Generic helper used for tests and potential future refactors
    def _calculate_device_score(self, model_size_gb: float, limit_gb: float) -> float:
        """Calculate normalized score versus device limit with improved scaling for different model sizes.

        Uses a more sophisticated scoring system that:
        - Provides better granularity for small models
        - Handles large models more appropriately
        - Uses exponential decay for better differentiation
        """
        if limit_gb <= 0:
            return 0.0
        
        ratio = model_size_gb / limit_gb
        
        # For very small models (under 10% of limit), give excellent score
        if ratio <= 0.1:
            return 1.0
        
        # For small models (10-50% of limit), use linear scaling with high scores
        elif ratio <= 0.5:
            # Linear interpolation from 1.0 to 0.9
            return 1.0 - (ratio - 0.1) * 0.25  # 0.1 -> 1.0, 0.5 -> 0.9
        
        # For medium models (50-100% of limit), use exponential decay
        elif ratio <= 1.0:
            # Exponential decay from 0.9 to 0.7
            return 0.9 * (1.0 - ratio) ** 0.5 + 0.7 * ratio
        
        # For large models (100-200% of limit), use steeper decay
        elif ratio <= 2.0:
            # Steep exponential decay from 0.7 to 0.2
            return 0.7 * (2.0 - ratio) ** 2 + 0.2 * (ratio - 1.0)
        
        # For very large models (over 200% of limit), use very steep decay
        elif ratio <= 4.0:
            # Very steep decay from 0.2 to 0.05
            return 0.2 * (4.0 - ratio) ** 3 + 0.05 * (ratio - 2.0)
        
        # For extremely large models (over 400% of limit), minimal score
        else:
            return max(0.0, 0.05 * (1.0 / ratio))

    async def _estimate_model_size(self, context: ModelContext) -> float:
        """Estimate model size with improved accuracy for different model types."""
        # extract from README content first (most reliable)
        if context.readme_content:
            size_from_readme = extract_model_size_from_text(context.readme_content)
            if size_from_readme:
                return size_from_readme

        # estimate from hugging face file information
        if context.hf_info and context.hf_info.get("files"):
            total_size_mb = 0
            files = context.hf_info["files"]
            
            # Handle both list and dict formats for files
            if isinstance(files, list):
                # If files is a list of file paths
                for file_path in files:
                    if file_path.endswith(".bin") or file_path.endswith(".safetensors"):
                        if "pytorch_model" in file_path:
                            # Use a reasonable default estimate for model files
                            total_size_mb += 1000  # Default 1GB per model file
                    elif file_path.endswith(".json"):
                        total_size_mb += 1  # config files are small
                    elif file_path.endswith(".txt"):
                        total_size_mb += 0.1  # text files are tiny
            elif isinstance(files, dict):
                # If files is a dict with file info
                for file_path, file_info in files.items():
                    if file_path.endswith(".bin") or file_path.endswith(".safetensors"):
                        if "pytorch_model" in file_path:
                            # Try to extract actual file size if available
                            file_size = file_info.get("size", 0) if isinstance(file_info, dict) else 0
                            if file_size > 0:
                                total_size_mb += file_size / (1024 * 1024)  # Convert bytes to MB
                            else:
                                # Use a reasonable default estimate for model files
                                total_size_mb += 1000  # Default 1GB per model file
                    elif file_path.endswith(".json"):
                        total_size_mb += 1  # config files are small
                    elif file_path.endswith(".txt"):
                        total_size_mb += 0.1  # text files are tiny

            if total_size_mb > 0:
                return total_size_mb / 1024.0  # Convert to GB

        # Generic size estimation based on model name patterns
        model_name = context.model_url.name.lower()
        
        # Use generic size indicators without hard-coding specific model types
        if "large" in model_name:
            return 4.0  # Generic large model
        elif "base" in model_name:
            return 1.0  # Generic base model
        elif "small" in model_name or "tiny" in model_name:
            return 0.5  # Generic small model
        elif "mini" in model_name:
            return 0.2  # Mini models
        elif "nano" in model_name:
            return 0.1  # Nano models

        # default assumption for unknown models
        return 2.0

