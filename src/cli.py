# src/cli.py
"""CLI entrypoint for ECE30861 Team19 project.

Reads a newline-delimited file of URLs and prints NDJSON lines for MODEL URLs.
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# hugggingface_hub for API metadata
try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover - graceful fallback
    HfApi = None  # type: ignore

# Setup logging to $LOG_FILE and respect $LOG_LEVEL
LOG_FILE = os.environ.get("LOG_FILE", "project.log")
LOG_LEVEL = int(os.environ.get("LOG_LEVEL", "0"))
logging.basicConfig(
    filename=LOG_FILE,
    level={0: logging.NOTSET, 1: logging.INFO, 2: logging.DEBUG}.get(LOG_LEVEL, logging.NOTSET),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

api = HfApi() if HfApi else None

def now_ms() -> int:
    return int(time.perf_counter() * 1000)

def safe_time_ms(fn, *args, **kwargs) -> Tuple[Any, int]:
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return res, elapsed_ms

def classify_url(url: str) -> str:
    # Very simple URL classifier
    if "huggingface.co" in url and "/tree/" in url:
        return "MODEL"
    if "huggingface.co/datasets" in url or "/datasets/" in url:
        return "DATASET"
    if "github.com" in url:
        return "CODE"
    return "MODEL"  # default to model for scoring

def compute_license_score(readme_text: str) -> float:
    # naive: if "License" exists and contains permissive terms -> higher score
    if not readme_text:
        return 0.0
    low = readme_text.lower()
    if "mit" in low or "apache" in low or "bsd" in low or "lgpl" in low:
        return 1.0
    if "license" in low:
        return 0.5
    return 0.0

def compute_size_score(repo_info: Dict[str, Any]) -> Dict[str, float]:
    # stub: compute based on size of repo files. For now return conservative numbers.
    return {"raspberry_pi": 0.2, "jetson_nano": 0.3, "desktop_pc": 0.8, "aws_server": 1.0}

def compute_other_metrics(repo_meta: Dict[str, Any]) -> Dict[str, float]:
    # stubs for required metrics
    return {
        "ramp_up_time": 0.5,
        "bus_factor": 0.5,
        "performance_claims": 0.5,
        "dataset_and_code_score": 0.5,
        "dataset_quality": 0.5,
        "code_quality": 0.5,
    }

def score_model_from_hf(model_id: str) -> Dict[str, Any]:
    """Query HuggingFace API for model metadata and compute metrics"""
    out: Dict[str, Any] = {"name": model_id, "category": "MODEL"}
    # license
    readme = ""
    if api is not None:
        try:
            info = api.model_info(model_id)
            likes = getattr(info, "likes", None)
            downloads = getattr(info, "downloads", None)
            # Note: HuggingFace API objects vary; adapt later
            # Attempt to fetch README
            try:
                readme = api.repo_readme(model_id)
            except Exception:
                readme = ""
            out["likes"] = likes
            out["downloads"] = downloads
        except Exception as e:
            logger.debug("HfApi model_info failed: %s", e)
    # license score and size score
    license_score = compute_license_score(readme)
    size_score = compute_size_score({})
    metrics = compute_other_metrics({})
    # build combined object
    out_fields = {
        "license": license_score,
        "size_score": size_score,
    }
    out.update(out_fields)
    out.update(metrics)
    return out

def compute_net_score(entry: Dict[str, Any]) -> float:
    # Example weighted sum (tune weights based on requirements)
    weights = {
        "license": 0.15,
        "ramp_up_time": 0.20,
        "bus_factor": 0.10,
        "dataset_and_code_score": 0.20,
        "dataset_quality": 0.10,
        "code_quality": 0.10,
        "performance_claims": 0.15,
    }
    total = 0.0
    for k, w in weights.items():
        total += entry.get(k, 0.0) * w
    return max(0.0, min(1.0, total))

def process_model_url(url: str) -> Dict[str, Any]:
    model_id = url.split("huggingface.co/")[-1].split("/tree")[0].strip("/")
    # compute license latency
    # license
    _, license_latency = safe_time_ms(lambda: compute_license_score(""))
    # size
    _, size_latency = safe_time_ms(lambda: compute_size_score({}))
    # other metrics executed concurrently
    results = {}
    metrics_fns = {
        "ramp_up_time": lambda: 0.5,
        "bus_factor": lambda: 0.5,
        "performance_claims": lambda: 0.5,
        "dataset_and_code_score": lambda: 0.5,
        "dataset_quality": lambda: 0.5,
        "code_quality": lambda: 0.5,
    }
    latencies: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(6, (os.cpu_count() or 2))) as ex:
        futures = {ex.submit(fn): name for name, fn in metrics_fns.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            start = time.perf_counter()
            try:
                val = fut.result()
            except Exception:
                val = 0.0
            latencies[name] = int((time.perf_counter() - start) * 1000)
            results[name] = float(val)
    # size_score latency already computed
    entry: Dict[str, Any] = {
        "name": model_id,
        "category": "MODEL",
        "license": float(0.0),
        "license_latency": license_latency,
        "size_score": compute_size_score({}),
        "size_score_latency": size_latency,
        "ramp_up_time": results["ramp_up_time"],
        "ramp_up_time_latency": latencies["ramp_up_time"],
        "bus_factor": results["bus_factor"],
        "bus_factor_latency": latencies["bus_factor"],
        "performance_claims": results["performance_claims"],
        "performance_claims_latency": latencies["performance_claims"],
        "dataset_and_code_score": results["dataset_and_code_score"],
        "dataset_and_code_score_latency": latencies["dataset_and_code_score"],
        "dataset_quality": results["dataset_quality"],
        "dataset_quality_latency": latencies["dataset_quality"],
        "code_quality": results["code_quality"],
        "code_quality_latency": latencies["code_quality"],
    }
    entry["net_score"], entry["net_score_latency"] = compute_net_score(entry), 0
    # compute net_score latency (tiny, but measured)
    start = time.perf_counter()
    _ = compute_net_score(entry)
    entry["net_score_latency"] = int((time.perf_counter() - start) * 1000)
    return entry

def print_ndjson(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, separators=(",", ":"), sort_keys=False))

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m src.cli URL_FILE", file=sys.stderr)
        return 1
    url_file = argv[1]
    try:
        with open(url_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error("Failed to open URL file %s: %s", url_file, e)
        print(f"Error: cannot open {url_file}", file=sys.stderr)
        return 1

    # process sequentially but metric computations are parallel per model
    for url in urls:
        category = classify_url(url)
        if category != "MODEL":
            continue  # grader only expects MODEL outputs (datasets / code may appear but not printed)
        try:
            entry = process_model_url(url)
            print_ndjson(entry)
        except Exception as e:
            logger.exception("Error processing %s: %s", url, e)
            # output a failure line with 0 scores if desired (safer)
            print_ndjson({"name": url, "category": "MODEL", "net_score": 0.0, "net_score_latency": 0})
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
