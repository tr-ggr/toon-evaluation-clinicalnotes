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

pred_values = _extract_text_values(pred)
ref_values = _extract_text_values(ref)

print(f"Pred text values: {len(pred_values)}")
print(f"Ref text values: {len(ref_values)}")
print(f"\nFirst 5 pred values:")
for v in list(pred_values)[:5]:
    print(f"  {v}")
print(f"\nFirst 5 ref values:")
for v in list(ref_values)[:5]:
    print(f"  {v}")

overlap = pred_values & ref_values
print(f"\nOverlap: {len(overlap)}")
print(f"Sample overlap:")
for v in list(overlap)[:5]:
    print(f"  {v}")
