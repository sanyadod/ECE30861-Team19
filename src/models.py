"""
Data models
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class URLCategory(str, Enum):
    """Categories for URLs."""

    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"


class ParsedURL(BaseModel):
    """Represents a parsed URL with metadata."""

    url: str
    category: URLCategory
    name: str
    platform: str  # e.g., "huggingface", "github"
    owner: Optional[str] = None
    repo: Optional[str] = None


class SizeScore(BaseModel):
    """Size score breakdown by device type."""

    raspberry_pi: float = Field(..., ge=0.0, le=1.0)
    jetson_nano: float = Field(..., ge=0.0, le=1.0)
    desktop_pc: float = Field(..., ge=0.0, le=1.0)
    aws_server: float = Field(..., ge=0.0, le=1.0)


class MetricResult(BaseModel):
    """Result of a single metric calculation."""

    score: Dict[str, float] | float #float = Field(..., ge=0.0, le=1.0)
    latency: int = Field(..., ge=0)  # milliseconds


class AuditResult(BaseModel):
    """Complete audit result for a model (NDJSON output format)."""

    name: str
    category: str = "MODEL"  # Always MODEL for output
    net_score: float = Field(..., ge=0.0, le=1.0)
    net_score_latency: int = Field(..., ge=0)

    ramp_up_time: float = Field(..., ge=0.0, le=1.0)
    ramp_up_time_latency: int = Field(..., ge=0)

    bus_factor: float = Field(..., ge=0.0, le=1.0)
    bus_factor_latency: int = Field(..., ge=0)

    performance_claims: float = Field(..., ge=0.0, le=1.0)
    performance_claims_latency: int = Field(..., ge=0)

    license: float = Field(..., ge=0.0, le=1.0)
    license_latency: int = Field(..., ge=0)

    size_score: SizeScore
    size_score_latency: int = Field(..., ge=0)

    dataset_and_code_score: float = Field(..., ge=0.0, le=1.0)
    dataset_and_code_score_latency: int = Field(..., ge=0)

    dataset_quality: float = Field(..., ge=0.0, le=1.0)
    dataset_quality_latency: int = Field(..., ge=0)

    code_quality: float = Field(..., ge=0.0, le=1.0)
    code_quality_latency: int = Field(..., ge=0)


class ModelContext(BaseModel):
    """Context for a model including associated datasets and code."""

    model_url: ParsedURL
    datasets: list[ParsedURL] = Field(default_factory=list)
    code_repos: list[ParsedURL] = Field(default_factory=list)

    # Cached data from API calls
    hf_info: Optional[Dict[str, Any]] = None
    readme_content: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
