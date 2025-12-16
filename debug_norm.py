#!/usr/bin/env python
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from toon_experiment.eval.run_eval import _extract_prediction, _extract_ground_truth, _normalize_to_schema
from toon_experiment.eval.metrics import _flatten

# Load first sample
output_path = Path("scripts/outputs/gemini-2.5-pro/json/sample_00000.json")
with output_path.open() as f:
    obj = json.load(f)

pred_raw = _extract_prediction(obj)
ref_raw = _extract_ground_truth(obj)

print("=== Before Normalization ===")
print(f"Pred keys: {set(pred_raw.keys())}")
print(f"Ref keys: {set(ref_raw.keys())}")

pred_norm = _normalize_to_schema(pred_raw)
ref_norm = _normalize_to_schema(ref_raw)

print("\n=== After Normalization ===")
print(f"Pred keys: {set(pred_norm.keys())}")
print(f"Ref keys: {set(ref_norm.keys())}")

print("\n=== Flattened After Normalization ===")
pred_flat = _flatten(pred_norm)
ref_flat = _flatten(ref_norm)

print(f"Pred flat (first 10):")
for k, v in list(pred_flat.items())[:10]:
    print(f"  {k}: {type(v).__name__}")

print(f"\nRef flat (first 10):")
for k, v in list(ref_flat.items())[:10]:
    print(f"  {k}: {type(v).__name__}")

print(f"\n Common keys: {set(pred_flat.keys()) & set(ref_flat.keys())}")
