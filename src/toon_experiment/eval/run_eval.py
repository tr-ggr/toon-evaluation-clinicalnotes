from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict, Any

from toon_experiment.eval.metrics import (
    bertscore_avg,
    field_precision_recall_f1,
    entity_array_f1,
    schema_coverage,
    parse_run_summary,
)
from toon_experiment.formats import json_format, yaml_format, toon_format
from toon_experiment.schemas.summary import Summary


FORMAT_MODULES = {
    "json": json_format,
    "yaml": yaml_format,
    "toon": toon_format,
}


def _load_preds(outputs_dir: Path) -> List[dict]:
    preds: List[dict] = []
    for path in sorted(outputs_dir.glob("sample_*.json")):
        with path.open("r", encoding="utf-8") as f:
            preds.append(json.load(f))
    return preds


def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block markers from text."""
    text = text.strip()
    # Remove opening ```language marker
    if text.startswith("```"):
        text = text[3:]
        # Skip language identifier (e.g., json, yaml, toon)
        if text and not text.startswith("\n"):
            text = text.lstrip("jsonhyamloot")  # Remove language identifier chars
        text = text.lstrip("\n")
    # Remove closing ```
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def _extract_prediction(obj: dict) -> dict:
    """Parse output_summary field to get prediction content."""
    if "output_summary" not in obj:
        # Fallback for old format
        return {k: v for k, v in obj.items() if k not in ["ground_truth_summary", "format"]}
    
    # Parse the output_summary based on format
    fmt = obj.get("format", "json")
    fmt_module = FORMAT_MODULES.get(fmt, json_format)
    
    try:
        output_text = _strip_markdown_code_block(obj["output_summary"])
        return fmt_module.loads(output_text)
    except Exception:
        # Fallback to empty dict on parse error
        return {}
    
    return {}


def _normalize_to_schema(obj: dict) -> dict:
    """Normalize a dict to match the Summary schema by validating and serializing through Pydantic.
    
    This ensures both predictions and references use consistent field names and structure.
    """
    try:
        # Validate against schema - this applies aliases and standardizes field names
        summary = Summary.model_validate(obj, from_attributes=False)
        # Serialize back to dict with normalized field names
        return summary.model_dump(mode="json", exclude_none=False, by_alias=True)
    except Exception:
        # If validation fails, return empty dict
        return {}


def _extract_ground_truth(obj: dict) -> dict:
    """Parse ground_truth_summary field to get reference content."""
    if "ground_truth_summary" not in obj:
        return {}
    
    # Parse the ground_truth_summary based on format
    fmt = obj.get("format", "json")
    fmt_module = FORMAT_MODULES.get(fmt, json_format)
    
    try:
        if isinstance(obj["ground_truth_summary"], dict):
            return obj["ground_truth_summary"]
        ground_truth_text = _strip_markdown_code_block(obj["ground_truth_summary"])
        return fmt_module.loads(ground_truth_text)
    except Exception:
        return {}


def _calculate_compression(output_objs: List[dict], limit: int = 10) -> Dict[str, Any]:
    """Calculate token compression metrics (output_summary size / full_note size).

    For Avg Input/Sample, load the ACN dataset and compute the average input
    characters using only the first ``limit`` samples (default 10), independent
    of how many output objects are available.

    Args:
        output_objs: List of parsed output objects with output_summary field.
        limit: Number of ACN samples to consider when computing input averages.

    Returns:
        Dict with compression ratios per sample and averages.
    """
    try:
        from toon_experiment.io.datasets import iter_acn_hf
    except ImportError:
        # If dataset not available, return empty compression metrics
        return {
            "total_input_chars": 0,
            "total_output_chars": 0,
            "average_input_chars": 0,
            "average_output_chars": 0,
            "compression_ratio": 0.0,
            "per_sample": [],
            "average_compression": 0.0,
        }
    
    # Load the first `limit` samples from ACN dataset for consistent indexing
    all_samples = list(iter_acn_hf(limit=limit))

    # Compute dataset-driven Avg Input/Sample strictly from the first `limit` notes
    if all_samples:
        dataset_avg_input = mean(len(s.full_note) for s in all_samples)
        dataset_total_input = sum(len(s.full_note) for s in all_samples)
    else:
        dataset_avg_input = 0
        dataset_total_input = 0
    
    compression_ratios = []
    total_input = 0
    total_output = 0
    per_sample_data = []
    
    for i, obj in enumerate(output_objs):
        if "output_summary" not in obj or i >= len(all_samples):
            continue
        
        sample = all_samples[i]
        
        # Get input size from original full_note
        input_len = len(sample.full_note)
        total_input += input_len
        
        # Get output size from output_summary (stripped of markdown)
        output_text = _strip_markdown_code_block(obj["output_summary"])
        output_len = len(output_text)
        total_output += output_len
        
        # Calculate compression ratio for this sample
        compression = output_len / input_len if input_len > 0 else 0.0
        compression_ratios.append(compression)
        per_sample_data.append({
            "sample": i,
            "input_chars": input_len,
            "output_chars": output_len,
            "compression": compression,
        })
    
    avg_compression = mean(compression_ratios) if compression_ratios else 0.0
    num_samples = len(per_sample_data)
    # For reporting, Avg Input/Sample is dataset-based; Avg Output/Sample is
    # computed from available outputs aligned to those samples.
    avg_input = dataset_avg_input
    avg_output = total_output / num_samples if num_samples > 0 else 0
    
    return {
        # Keep totals as observed from paired samples for compression math
        "total_input_chars": total_input,
        "total_output_chars": total_output,
        "average_input_chars": avg_input,
        "average_output_chars": avg_output,
        "compression_ratio": total_output / total_input if total_input > 0 else 0.0,
        "per_sample": per_sample_data,
        "average_compression": avg_compression,
    }


def evaluate(outputs_dir: Path) -> Dict[str, Any]:
    """Evaluate parsed outputs against ground truth summaries.

    Args:
        outputs_dir: Directory containing sample_*.json parsed outputs.

    Returns:
        Dict containing all evaluation metrics including field-level, entity-level,
        coverage, format comparison metrics, and compression metrics.
    """
    output_objs = _load_preds(outputs_dir)
    preds = [_extract_prediction(obj) for obj in output_objs]
    refs = [_extract_ground_truth(obj) for obj in output_objs]
    
    paired = list(zip(preds, refs))
    
    # Field-level metrics
    p_vals: List[float] = []
    r_vals: List[float] = []
    f1_vals: List[float] = []
    bert_vals: List[float] = []
    
    # Entity-level metrics (per array field)
    entity_metrics: Dict[str, List[Tuple[float, float, float]]] = {}
    
    # Coverage metrics
    coverage_vals: List[Dict[str, float]] = []
    
    for pred, ref in paired:
        # Field-level
        p, r, f1 = field_precision_recall_f1(pred, ref)
        p_vals.append(p)
        r_vals.append(r)
        f1_vals.append(f1)
        bert_vals.append(bertscore_avg(pred, ref))
        
        # Entity-level for arrays
        entity_scores = entity_array_f1(pred, ref)
        for field, (ep, er, ef1) in entity_scores.items():
            if field not in entity_metrics:
                entity_metrics[field] = []
            entity_metrics[field].append((ep, er, ef1))
        
        # Coverage
        coverage = schema_coverage(pred, ref)
        coverage_vals.append(coverage)
    
    # Aggregate entity metrics
    entity_aggregated = {}
    for field, scores in entity_metrics.items():
        if scores:
            ps, rs, f1s = zip(*scores)
            entity_aggregated[field] = {
                "precision": mean(ps),
                "recall": mean(rs),
                "f1": mean(f1s),
            }
    
    # Aggregate coverage metrics
    coverage_aggregated = {}
    if coverage_vals:
        for section in coverage_vals[0].keys():
            section_vals = [cv[section] for cv in coverage_vals]
            coverage_aggregated[section] = mean(section_vals)
    
    # Parse run summary for format comparison metrics
    run_summary_path = outputs_dir / "run_summary.txt"
    format_metrics = parse_run_summary(run_summary_path)
    
    # Calculate compression metrics
    compression_metrics = _calculate_compression(output_objs)
    
    return {
        # Field-level metrics
        "field_precision": mean(p_vals) if p_vals else 0.0,
        "field_recall": mean(r_vals) if r_vals else 0.0,
        "field_f1": mean(f1_vals) if f1_vals else 0.0,
        "bertscore_avg": mean(bert_vals) if bert_vals else 0.0,
        
        # Entity-level metrics
        "entity_metrics": entity_aggregated,
        
        # Coverage metrics
        "coverage": coverage_aggregated,
        
        # Format comparison metrics
        "format_metrics": format_metrics,
        
        # Compression metrics
        "compression": compression_metrics,
    }
