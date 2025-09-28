"""
Data models  used across parsing and scoring
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class URLCategory(str, Enum):
    #coarse buckets, add more if we treat HF scpaes, etc differently

    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"


class ParsedURL(BaseModel):
 #minimal information needed to reason about a URL without hitting the network

    url: str
    category: URLCategory
    name: str
    platform: str  # e.g., "huggingface", "github"
    owner: Optional[str] = None
    repo: Optional[str] = None


class SizeScore(BaseModel):
    #per device, the normalized scores in the format of [0,1]

    raspberry_pi: float = Field(..., ge=0.0, le=1.0)
    jetson_nano: float = Field(..., ge=0.0, le=1.0)
    desktop_pc: float = Field(..., ge=0.0, le=1.0)
    aws_server: float = Field(..., ge=0.0, le=1.0)


class MetricResult(BaseModel):
    #single metric output, most of them return a scalar in [0,1]

    score: float = Field(..., ge=0.0, le=1.0)
    latency: int = Field(..., ge=0)  # milliseconds unit


class AuditResult(BaseModel):
    #it is flattened view for NDJSON, keeping the normalized numbers 

    name: str
    category: str = "MODEL"  # Always MODEL for output is expected 
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
    #all we know about the model + related assets, no netwrok required here 

    model_url: ParsedURL
    datasets: list[ParsedURL] = Field(default_factory=list)
    code_repos: list[ParsedURL] = Field(default_factory=list)

    # Cached data from API calls
    hf_info: Optional[Dict[str, Any]] = None
    readme_content: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
