from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from bert_score import score as bert_score


def _is_empty(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, list) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False


def _normalize(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _flatten(obj: object, prefix: str = "") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten(v, child_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]"
            flat.update(_flatten(v, child_prefix))
    else:
        flat[prefix] = obj
    return flat


def field_precision_recall_f1(pred: Dict, ref: Dict) -> Tuple[float, float, float]:
    pflat = _flatten(pred)
    rflat = _flatten(ref)
    tp = fp = fn = 0
    for key, r_val in rflat.items():
        if key == "schema_version":
            continue
        p_val = pflat.get(key)
        if _is_empty(r_val):
            # ignore empty reference fields
            continue
        if _is_empty(p_val):
            fn += 1
            continue
        if _normalize(p_val) == _normalize(r_val):
            tp += 1
        else:
            fn += 1
            fp += 1
    # extras in prediction not in reference and non-empty
    for key, p_val in pflat.items():
        if key not in rflat and not _is_empty(p_val):
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def bertscore_avg(pred: Dict, ref: Dict, lang: str = "en") -> float:
    pairs: List[Tuple[str, str]] = []
    pflat = _flatten(pred)
    rflat = _flatten(ref)
    for key, r_val in rflat.items():
        if key == "schema_version":
            continue
        p_val = pflat.get(key)
        if isinstance(p_val, str) and isinstance(r_val, str):
            if _is_empty(p_val) or _is_empty(r_val):
                continue
            pairs.append((p_val, r_val))
    if not pairs:
        return 0.0
    cands, refs = zip(*pairs)
    _, _, f1 = bert_score(list(cands), list(refs), lang=lang, rescale_with_baseline=True)
    return float(np.mean(f1.numpy()))
