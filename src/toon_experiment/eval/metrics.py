from __future__ import annotations

from typing import Dict, List, Tuple, Set, Any
from pathlib import Path
import re

import numpy as np
from bert_score import score as bert_score
from difflib import SequenceMatcher


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


def _extract_text_values(obj: object) -> Set[str]:
    """Extract all non-empty string values from nested structure.
    
    Filters out:
    - Strings with 3 chars or less
    - The string "none" (case-insensitive)
    - Empty or None values
    """
    values: Set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            values.update(_extract_text_values(v))
    elif isinstance(obj, list):
        for item in obj:
            values.update(_extract_text_values(item))
    elif isinstance(obj, str) and obj.strip():
        normalized = _normalize(obj)
        # Skip very short values, "none" strings, and pure numbers
        if len(normalized) > 3 and normalized != "none":
            values.add(normalized)
    return values


def field_precision_recall_f1(pred: Dict, ref: Dict) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1 using flattened field matching with normalization.
    
    Compares the flattened dictionaries but uses normalized string comparison
    to be more lenient with whitespace and casing differences.
    
    Fixed: Mismatches now only count as FN (not both FP and FN) to avoid double penalty.
    """
    pflat = _flatten(pred)
    rflat = _flatten(ref)
    
    tp = fp = fn = 0
    
    # For each reference field
    for key, r_val in rflat.items():
        if key == "schema_version":
            continue
        
        if _is_empty(r_val):
            # Ignore empty reference fields
            continue
        
        # Look for matching prediction field
        p_val = pflat.get(key)
        
        if _is_empty(p_val):
            # Missing in prediction
            fn += 1
            continue
        
        # Compare normalized values
        if _normalize(p_val) == _normalize(r_val):
            tp += 1
        else:
            # Mismatch: count only as FN (value present but wrong)
            fn += 1
    
    # Count extra predictions not in reference
    for key, p_val in pflat.items():
        if key not in rflat and not _is_empty(p_val):
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
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


def _fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Check if two strings are fuzzy matches using sequence matcher."""
    return SequenceMatcher(None, _normalize(s1), _normalize(s2)).ratio() >= threshold


def _entity_to_string(entity: Any) -> str:
    """Convert an entity (dict or string) to a comparable string representation."""
    if isinstance(entity, dict):
        # Create signature from non-empty values
        parts = []
        for k, v in sorted(entity.items()):
            if not _is_empty(v):
                parts.append(f"{k}:{_normalize(v)}")
        return "|".join(parts)
    return _normalize(str(entity))


def _find_best_match(entity: Any, candidates: List[Any], threshold: float = 0.7) -> Tuple[int, float]:
    """Find the best matching entity from candidates using fuzzy matching.
    
    Returns:
        Tuple of (index, score) where index=-1 if no match above threshold
    """
    entity_str = _entity_to_string(entity)
    best_idx = -1
    best_score = 0.0
    
    for idx, candidate in enumerate(candidates):
        candidate_str = _entity_to_string(candidate)
        score = SequenceMatcher(None, entity_str, candidate_str).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
    
    if best_score >= threshold:
        return best_idx, best_score
    return -1, 0.0


def entity_array_f1(
    pred: Dict,
    ref: Dict,
    array_fields: List[str] = None,
    threshold: float = 0.7
) -> Dict[str, Tuple[float, float, float]]:
    """Calculate entity-level precision, recall, F1 for array fields.
    
    Uses fuzzy matching to handle variations in entity representation.
    
    Args:
        pred: Prediction dictionary
        ref: Reference dictionary
        array_fields: List of array field names to evaluate. If None, uses default clinical arrays.
        threshold: Fuzzy matching threshold (0.0-1.0)
    
    Returns:
        Dict mapping field name to (precision, recall, f1)
    """
    if array_fields is None:
        array_fields = [
            "symptoms",
            "treatments",
            "diagnosis tests",
            "medical examinations",
            "surgeries",
            "admission"
        ]
    
    results = {}
    
    for field in array_fields:
        pred_entities = pred.get(field, [])
        ref_entities = ref.get(field, [])
        
        if not isinstance(pred_entities, list):
            pred_entities = []
        if not isinstance(ref_entities, list):
            ref_entities = []
        
        if len(ref_entities) == 0:
            # No reference entities
            if len(pred_entities) == 0:
                results[field] = (1.0, 1.0, 1.0)
            else:
                results[field] = (0.0, 1.0, 0.0)
            continue
        
        if len(pred_entities) == 0:
            # Missing all predictions
            results[field] = (0.0, 0.0, 0.0)
            continue
        
        # Match entities using fuzzy matching
        ref_matched = [False] * len(ref_entities)
        pred_matched = [False] * len(pred_entities)
        
        for p_idx, pred_entity in enumerate(pred_entities):
            # Find best match in ref
            match_idx, score = _find_best_match(
                pred_entity,
                [r for r_idx, r in enumerate(ref_entities) if not ref_matched[r_idx]],
                threshold
            )
            
            if match_idx >= 0:
                # Map back to original index
                unmatched_indices = [i for i, matched in enumerate(ref_matched) if not matched]
                actual_idx = unmatched_indices[match_idx]
                ref_matched[actual_idx] = True
                pred_matched[p_idx] = True
        
        tp = sum(pred_matched)
        fp = len(pred_entities) - tp
        fn = len(ref_entities) - sum(ref_matched)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[field] = (precision, recall, f1)
    
    return results


def schema_coverage(pred: Dict, ref: Dict = None) -> Dict[str, float]:
    """Calculate field population rate for major schema sections.
    
    Args:
        pred: Prediction dictionary
        ref: Reference dictionary (optional, not used but kept for API consistency)
    
    Returns:
        Dict mapping section name to population rate (0.0-1.0)
    """
    sections = {
        "visit motivation": ["visit motivation"],
        "admission": ["admission"],
        "patient information": ["patient information"],
        "patient medical history": ["patient medical history"],
        "surgeries": ["surgeries"],
        "symptoms": ["symptoms"],
        "medical examinations": ["medical examinations"],
        "diagnosis tests": ["diagnosis tests"],
        "treatments": ["treatments"],
        "discharge": ["discharge"],
    }
    
    coverage = {}
    
    for section_name, field_paths in sections.items():
        populated = False
        for field_path in field_paths:
            value = pred.get(field_path)
            if not _is_empty(value):
                # For arrays, check if non-empty and has non-empty elements
                if isinstance(value, list):
                    if len(value) > 0:
                        # Check if at least one element has non-empty content
                        for item in value:
                            if isinstance(item, dict):
                                if any(not _is_empty(v) for v in item.values()):
                                    populated = True
                                    break
                            elif not _is_empty(item):
                                populated = True
                                break
                # For dicts, check if has non-empty values
                elif isinstance(value, dict):
                    if any(not _is_empty(v) for v in value.values()):
                        populated = True
                # For scalars, check if non-empty
                else:
                    populated = True
            
            if populated:
                break
        
        coverage[section_name] = 1.0 if populated else 0.0
    
    return coverage


def parse_run_summary(summary_path: Path) -> Dict[str, Any]:
    """Parse run_summary.txt file to extract format comparison metrics.
    
    Args:
        summary_path: Path to run_summary.txt file
    
    Returns:
        Dict with keys: success_rate, avg_retries, avg_time, total_samples, failed_samples
    """
    if not summary_path.exists():
        return {
            "success_rate": 0.0,
            "avg_retries": 0.0,
            "avg_time": 0.0,
            "total_samples": 0,
            "failed_samples": 0
        }
    
    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract summary statistics from the TOTAL line
    total_match = re.search(r"TOTAL\s+(\d+)/(\d+)\s+\d+\s+(\d+)\s+([\d.]+)", content)
    avg_match = re.search(r"AVERAGE\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", content)
    
    if total_match and avg_match:
        succeeded = int(total_match.group(1))
        total = int(total_match.group(2))
        avg_attempts = float(avg_match.group(1))
        avg_retries = float(avg_match.group(2))
        avg_time = float(avg_match.group(3))
        
        return {
            "success_rate": succeeded / total if total > 0 else 0.0,
            "avg_retries": avg_retries,
            "avg_time": avg_time,
            "total_samples": total,
            "failed_samples": total - succeeded
        }
    
    return {
        "success_rate": 0.0,
        "avg_retries": 0.0,
        "avg_time": 0.0,
        "total_samples": 0,
        "failed_samples": 0
    }
