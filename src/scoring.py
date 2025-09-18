"""
Basic scoring system placeholder.
"""
import asyncio
from typing import Dict, Any

from .models import ModelContext, AuditResult, SizeScore
from .utils import measure_time
from .logging_utils import get_logger

logger = get_logger()


class MetricScorer:
    """Minimal scorer: all metrics return 0.0."""

    def __init__(self, config_path: str = None):
        # ignore config for now
        self.config = {}

    async def score_model(self, context: ModelContext) -> AuditResult:
        """Return all metrics as 0.0."""
        with measure_time() as get_latency:
            # Simulate “metric results” with all zeroes
            results: Dict[str, Any] = {
                "ramp_up_time": 0.0,
                "bus_factor": 0.0,
                "performance_claims": 0.0,
                "license": 0.0,
                "size_score": SizeScore(
                    raspberry_pi=0.0,
                    jetson_nano=0.0,
                    desktop_pc=0.0,
                    aws_server=0.0,
                ),
                "dataset_and_code_score": 0.0,
                "dataset_quality": 0.0,
                "code_quality": 0.0,
            }

            net_score = 0.0

        return AuditResult(
            name=context.model_url.name,
            category="MODEL",
            net_score=net_score,
            net_score_latency=get_latency(),

            ramp_up_time=results["ramp_up_time"],
            ramp_up_time_latency=0,

            bus_factor=results["bus_factor"],
            bus_factor_latency=0,

            performance_claims=results["performance_claims"],
            performance_claims_latency=0,

            license=results["license"],
            license_latency=0,

            size_score=results["size_score"],
            size_score_latency=0,

            dataset_and_code_score=results["dataset_and_code_score"],
            dataset_and_code_score_latency=0,

            dataset_quality=results["dataset_quality"],
            dataset_quality_latency=0,

            code_quality=results["code_quality"],
            code_quality_latency=0,
        )
