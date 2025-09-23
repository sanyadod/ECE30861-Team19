"""
Parallel metric computation and scoring system.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List
import yaml
from pathlib import Path

from .models import ModelContext, AuditResult, MetricResult, SizeScore
from .metrics.ramp_up import RampUpTimeMetric
from .metrics.bus_factor import BusFactorMetric
from .metrics.performance_claims import PerformanceClaimsMetric
from .metrics.license_score import LicenseScoreMetric
from .metrics.size_score import SizeScoreMetric
from .metrics.dataset_and_code import DatasetAndCodeScoreMetric
from .metrics.dataset_quality import DatasetQualityMetric
from .metrics.code_quality import CodeQualityMetric
from .hf_api import HuggingFaceAPI
from .utils import measure_time
from .logging_utils import get_logger


logger = get_logger()


class MetricScorer:
    """Handles parallel metric computation and scoring."""
    
    def __init__(self, config_path: str = "config/weights.yaml"):
        self.config = self._load_config(config_path)
        self.metrics = [
            RampUpTimeMetric(),
            BusFactorMetric(),
            PerformanceClaimsMetric(),
            LicenseScoreMetric(),
            SizeScoreMetric(),
            DatasetAndCodeScoreMetric(),
            DatasetQualityMetric(),
            CodeQualityMetric(),
        ]
        self.hf_api = HuggingFaceAPI()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'metric_weights': {
                'ramp_up_time': 0.15,
                'bus_factor': 0.10,
                'performance_claims': 0.15,
                'license': 0.10,
                'size_score': 0.15,
                'dataset_and_code_score': 0.15,
                'dataset_quality': 0.10,
                'code_quality': 0.10,
            },
            'thresholds': {}
        }
    
    async def score_model(self, context: ModelContext) -> AuditResult:
        """Score a model using all metrics in parallel."""
        # Enrich context with API data
        await self._enrich_context(context)
        
        # Compute all metrics in parallel
        with measure_time() as get_net_latency:
            metric_results = await self._compute_metrics_parallel(context)
            
            # Calculate net score
            net_score = self._calculate_net_score(metric_results)
        
        # Build audit result
        size_score_result = metric_results.get('size_score')
        size_score_obj = size_score_result if isinstance(size_score_result, SizeScore) else SizeScore(
            raspberry_pi=0.0, jetson_nano=0.0, desktop_pc=0.0, aws_server=0.0
        )
        
        return AuditResult(
            name=context.model_url.name,
            category="MODEL",
            net_score=net_score,
            net_score_latency=get_net_latency(),
            
            ramp_up_time=metric_results['ramp_up_time'].score,
            ramp_up_time_latency=metric_results['ramp_up_time'].latency,
            
            bus_factor=metric_results['bus_factor'].score,
            bus_factor_latency=metric_results['bus_factor'].latency,
            
            performance_claims=metric_results['performance_claims'].score,
            performance_claims_latency=metric_results['performance_claims'].latency,
            
            license=metric_results['license'].score,
            license_latency=metric_results['license'].latency,
            
            size_score=size_score_obj,
            size_score_latency=metric_results['size_score_latency'],
            
            dataset_and_code_score=metric_results['dataset_and_code_score'].score,
            dataset_and_code_score_latency=metric_results['dataset_and_code_score'].latency,
            
            dataset_quality=metric_results['dataset_quality'].score,
            dataset_quality_latency=metric_results['dataset_quality'].latency,
            
            code_quality=metric_results['code_quality'].score,
            code_quality_latency=metric_results['code_quality'].latency,
        )
    
    async def _enrich_context(self, context: ModelContext):
        """Enrich context with data from APIs."""
        # Get HF model info
        context.hf_info = await self.hf_api.get_model_info(context.model_url)
        
        # Get README content
        context.readme_content = await self.hf_api.get_readme_content(context.model_url)
        
        # Get model config
        context.config_data = await self.hf_api.get_model_config(context.model_url)
        
        logger.info(f"Enriched context for {context.model_url.name}")
    
    async def _compute_metrics_parallel(self, context: ModelContext) -> Dict[str, Any]:
        """Compute all metrics in parallel."""
        # Create tasks for all metrics
        tasks = []
        for metric in self.metrics:
            if metric.name == 'size_score':
                # Size score returns a SizeScore object, handle specially
                task = self._compute_size_metric_with_latency(metric, context)
            else:
                task = metric.compute(context, self.config)
            tasks.append((metric.name, task))
        
        # Execute all tasks concurrently
        results = {}
        for metric_name, task in tasks:
            try:
                result = await task
                if metric_name == 'size_score':
                    # Special handling for size score which returns tuple
                    if isinstance(result, tuple):
                        size_scores, latency = result
                        results[metric_name] = size_scores
                        results['size_score_latency'] = latency
                    else:
                        results[metric_name] = result
                        results['size_score_latency'] = 0
                else:
                    results[metric_name] = result
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                # Provide default result
                if metric_name == 'size_score':
                    results[metric_name] = SizeScore(
                        raspberry_pi=0.0, jetson_nano=0.0, desktop_pc=0.0, aws_server=0.0
                    )
                    results['size_score_latency'] = 0
                else:
                    results[metric_name] = MetricResult(score=0.0, latency=0)
        
        return results
    
    async def _compute_size_metric_with_latency(self, metric, context: ModelContext) -> tuple:
        """Special handling for size score metric."""
        with measure_time() as get_latency:
            size_scores = await metric._calculate_size_scores(context, self.config)
        
        # Return both size scores and latency
        return size_scores, get_latency()
    
    def _calculate_net_score(self, metric_results: Dict[str, Any]) -> float:
        """Calculate weighted net score from individual metrics."""
        weights = self.config.get('metric_weights', {})
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, result in metric_results.items():
            if metric_name == 'size_score' or metric_name.endswith('_latency'):
                continue  # Skip size_score object and latency fields
            
            weight = weights.get(metric_name, 0.0)
            if isinstance(result, MetricResult):
                total_score += result.score * weight
                total_weight += weight
        
        # Handle size score specially (average across devices)
        size_score = metric_results.get('size_score')
        if isinstance(size_score, SizeScore):
            size_weight = weights.get('size_score', 0.0)
            avg_size_score = (size_score.raspberry_pi + size_score.jetson_nano + 
                             size_score.desktop_pc + size_score.aws_server) / 4.0
            total_score += avg_size_score * size_weight
            total_weight += size_weight
        
        return total_score / max(total_weight, 1.0)