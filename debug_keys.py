#!/usr/bin/env python
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from toon_experiment.eval.run_eval import _extract_prediction, _extract_ground_truth
from toon_experiment.eval.metrics import _extract_text_values

# Load first sample
output_path = Path("scripts/outputs/gemini-2.5-pro/json/sample_00000.json")
with output_path.open() as f:
    obj = json.load(f)

pred = _extract_prediction(obj)
ref = _extract_ground_truth(obj)

print("=== Keys in Prediction ===")
print(list(pred.keys()))

print("\n=== Keys in Reference ===")
print(list(ref.keys()))

pred_values = sorted(_extract_text_values(pred))
ref_values = sorted(_extract_text_values(ref))

print(f"\n=== Prediction Values ({len(pred_values)}) ===")
for v in pred_values[:10]:
    print(f"  {v[:60]}...")

print(f"\n=== Reference Values ({len(ref_values)}) ===")
for v in ref_values:
    print(f"  {v[:60]}...")

overlap = set(pred_values) & set(ref_values)
print(f"\n=== Exact Overlap ({len(overlap)}) ===")
for v in overlap:
    print(f"  {v}")
