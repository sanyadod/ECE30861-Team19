from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
from ..models import ModelContext, MetricResult
from ..utils import measure_time
from ..logging_utils import get_logger

logger = get_logger()


class Metric(ABC):

    name: str

    @abstractmethod
    async def compute(self, context: ModelContext) -> MetricResult:
        
        raise NotImplementedError

    async def _timed(self, fn, *args, **kwargs) -> Tuple[float, int]:
        with measure_time() as get_latency:
            value = await fn(*args, **kwargs)
            latency_ms = get_latency()
        return value, latency_ms
