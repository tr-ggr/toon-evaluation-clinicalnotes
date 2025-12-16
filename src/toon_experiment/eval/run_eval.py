from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List, Tuple

from toon_experiment.eval.metrics import bertscore_avg, field_precision_recall_f1
from toon_experiment.io.datasets import iter_acn_hf


def _load_preds(outputs_dir: Path) -> List[dict]:
    preds: List[dict] = []
    for path in sorted(outputs_dir.glob("sample_*.json")):
        with path.open("r", encoding="utf-8") as f:
            preds.append(json.load(f))
    return preds


def _extract_prediction(obj: dict) -> dict:
    """Return the prediction content, stripping metadata if present."""
    if isinstance(obj, dict) and "prediction" in obj and isinstance(obj["prediction"], dict):
        return obj["prediction"]
    return {k: v for k, v in obj.items() if k != "ground_truth_summary"}


def evaluate(outputs_dir: Path, limit: int | None = None) -> Tuple[float, float, float, float]:
    """Evaluate parsed outputs against ACN summaries from Hugging Face.

    Args:
        outputs_dir: Directory containing sample_*.json parsed outputs.
        limit: Maximum number of samples to evaluate.

    Returns:
        Tuple of (precision, recall, f1, bertscore) averaged across samples.
    """
    preds = [_extract_prediction(p) for p in _load_preds(outputs_dir)]
    refs = [s.summary for s in iter_acn_hf(limit=limit)]
    paired = list(zip(preds, refs))
    p_vals: List[float] = []
    r_vals: List[float] = []
    f1_vals: List[float] = []
    bert_vals: List[float] = []
    for pred, ref in paired:
        p, r, f1 = field_precision_recall_f1(pred, ref)
        p_vals.append(p)
        r_vals.append(r)
        f1_vals.append(f1)
        bert_vals.append(bertscore_avg(pred, ref))
    return (
        mean(p_vals) if p_vals else 0.0,
        mean(r_vals) if r_vals else 0.0,
        mean(f1_vals) if f1_vals else 0.0,
        mean(bert_vals) if bert_vals else 0.0,
    )
