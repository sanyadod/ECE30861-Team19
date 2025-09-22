# to measure : is it clearly referencing a dataset, does the repo include runnable example code
# scoring: dataset_and_code_score = 0.5 * has_dataset + 0.5 * has_example_code
#          dataset_and_code_score_latency = time to compute (ms)

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from models import MetricResult

_DATASET_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/[A-Za-z0-9_.\-/]+",
    re.IGNORECASE,
)

class DatasetAndCodeAvailabilityMetric:
    NAME = "dataset_and_code_score"

    def _has_dataset_link(
        self,
        readme_text: str,
        model_index_json: Optional[dict],
    ) -> bool:
        """
        Returns True if any reasonable signal of a dataset reference is found.
        Signals (any one is enough):
          1) model_index.json contains 'dataset'/'datasets'
          2) README has explicit HF dataset URL
          3) README mentions dataset-like keywords
        """
        if model_index_json:
            blob = str(model_index_json).lower()
            if "dataset" in blob or "datasets" in blob:
                return True

        if readme_text and _DATASET_URL_RE.search(readme_text):
            return True

        if readme_text and re.search(
            r"\b(dataset|datasets|training data|corpus)\b", readme_text, re.IGNORECASE
        ):
            return True

        return False

    def _has_example_code(self, repo_dir: Path, readme_text: str) -> bool:
        if (repo_dir / "examples").exists() or (repo_dir / "notebooks").exists() or (repo_dir / "scripts").exists():
            return True

        for fn in ("train.py", "finetune.py", "eval.py", "inference.py"):
            if (repo_dir / fn).exists():
                return True

        if readme_text and re.search(
            r"(^|\n)\s*```(?:bash|sh|python)[\s\S]*?(python\s+[^\n]*\.py)",
            readme_text,
            re.IGNORECASE,
        ):
            return True

        return False

    async def run(self, context) -> MetricResult:
        t0 = time.time()

        readme_text: str = getattr(context, "readme_text", "") or ""
        model_index_json = getattr(context, "model_index_json", None)
        repo_dir = Path(getattr(context, "repo_dir"))

        has_dataset = self._has_dataset_link(readme_text, model_index_json)
        has_example = self._has_example_code(repo_dir, readme_text)

        score = 0.5 * int(has_dataset) + 0.5 * int(has_example)
        latency_ms = int((time.time() - t0) * 1000)

        return MetricResult(
            name=self.NAME,
            score=float(score),
            latency_ms=latency_ms,
            details={"has_dataset": has_dataset, "has_example_code": has_example},
        )
