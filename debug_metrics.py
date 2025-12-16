#!/usr/bin/env python
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from toon_experiment.eval.run_eval import _extract_prediction, _extract_ground_truth
from toon_experiment.eval.metrics import field_precision_recall_f1, _flatten

# Load first sample
output_path = Path("scripts/outputs/gemini-2.5-pro/json/sample_00000.json")
with output_path.open() as f:
    obj = json.load(f)

pred = _extract_prediction(obj)
ref = _extract_ground_truth(obj)

print("=== Flattened Prediction ===")
pred_flat = _flatten(pred)
for k, v in list(pred_flat.items())[:10]:
    print(f"  {k}: {v}")
print(f"  ... ({len(pred_flat)} total keys)")

print("\n=== Flattened Reference ===")
ref_flat = _flatten(ref)
for k, v in list(ref_flat.items())[:10]:
    print(f"  {k}: {v}")
print(f"  ... ({len(ref_flat)} total keys)")

print("\n=== Metrics ===")
p, r, f1 = field_precision_recall_f1(pred, ref)
print(f"Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

# Check key matching
print("\n=== Key Matching Analysis ===")
print(f"Prediction keys: {set(pred_flat.keys())}")
print(f"Reference keys: {set(ref_flat.keys())}")
print(f"Common keys: {set(pred_flat.keys()) & set(ref_flat.keys())}")
print(f"Only in pred: {set(pred_flat.keys()) - set(ref_flat.keys())}")
print(f"Only in ref: {set(ref_flat.keys()) - set(pred_flat.keys())}")
