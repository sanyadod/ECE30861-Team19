from __future__ import annotations
import re
import time
from typing import Dict, List, Optional
from huggingface_hub import HfApi
from models import MetricResult

_DATASET_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)",
    re.IGNORECASE,
)

_FIELDS = [
    "name_desc", "license", "size_samples", "tasks",
    "splits", "provenance", "ethics", "load_instructions"
]

class DatasetQualityMetric:
    NAME = "dataset_quality"

    def __init__(self, hf_api: Optional[HfApi] = None):
        self.api = hf_api or HfApi()

    def _extract_ids(self, readme_text: str) -> List[str]:
        ids = [f"{m.group(1)}/{m.group(2)}" for m in _DATASET_URL_RE.finditer(readme_text or "")]
        return list(dict.fromkeys(ids))

    def _hits_from_card(self, card: Dict) -> int:
        hits = 0
        if any(card.get(k) for k in ("pretty_name", "title", "description")): hits += 1
        if card.get("license"): hits += 1
        if any(card.get(k) for k in ("size", "num_examples", "num_rows")): hits += 1
        if card.get("task_categories") or card.get("task_ids"): hits += 1
        if card.get("splits"): hits += 1
        if card.get("source_datasets") or card.get("citation"): hits += 1
        if card.get("ethical_considerations"): hits += 1
        if card.get("usage") or card.get("card_data") or card.get("configs"): hits += 1
        return hits

    def _hits_from_readme(self, text: str) -> int:
        hits = 0
        if re.search(r"\b(dataset|corpus|training data)\b", text, re.I): hits += 1
        if re.search(r"\blicense\b", text, re.I): hits += 1
        if re.search(r"\b\d[\d,]*\s*(samples|examples|instances)\b", text, re.I): hits += 1
        if re.search(r"\b(task|benchmark)\b", text, re.I): hits += 1
        if re.search(r"\b(train|validation|test)\b", text, re.I): hits += 1
        if re.search(r"\bsource|provenance|citation\b", text, re.I): hits += 1
        if re.search(r"\bethic|risk|bias\b", text, re.I): hits += 1
        if re.search(r"\b(load|from\s+datasets\s+import|pip install datasets)\b", text, re.I): hits += 1
        return hits

    async def run(self, context) -> MetricResult:
        t0 = time.time()
        readme_text = getattr(context, "readme_text", "") or ""
        details: Dict = {}

        candidates = self._extract_ids(readme_text)
        best_hits = -1
        best_id = None
        last_error = None

        for ds_id in candidates:
            try:
                info = self.api.dataset_info(ds_id)
                card_data = info.card_data or {}
                hits = self._hits_from_card(card_data)
                if hits > best_hits:
                    best_hits = hits
                    best_id = ds_id
            except Exception as e:
                last_error = str(e)

        if best_hits >= 0:
            hits = best_hits
            details["dataset_id"] = best_id
            if last_error:
                details["note"] = "Some dataset cards failed to fetch"
        else:
            hits = self._hits_from_readme(readme_text)
            if candidates:
                details["warning"] = "Failed to fetch HF dataset cards; used README heuristics"

        score = hits / len(_FIELDS)
        latency_ms = int((time.time() - t0) * 1000)

        return MetricResult(
            name=self.NAME,
            score=float(score),
            latency_ms=latency_ms,
            details=details or None,
        )
