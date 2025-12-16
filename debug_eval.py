#!/usr/bin/env python
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from toon_experiment.eval.run_eval import _strip_markdown_code_block, _extract_prediction, _extract_ground_truth

# Load first sample
output_path = Path("scripts/outputs/gemini-2.5-pro/json/sample_00000.json")
with output_path.open() as f:
    obj = json.load(f)

print("Keys in object:", obj.keys())
print("\nFormat:", obj.get("format"))
print("\nFirst 100 chars of output_summary:")
print(obj["output_summary"][:100])
print("\nFirst 100 chars of ground_truth_summary:")
print(obj["ground_truth_summary"][:100])

print("\n\n=== Testing _extract_prediction ===")
pred = _extract_prediction(obj)
print("Prediction keys:", pred.keys() if pred else "EMPTY")
print("Prediction sample (first 200 chars of str):", str(pred)[:200])

print("\n\n=== Testing _extract_ground_truth ===")
ref = _extract_ground_truth(obj)
print("Reference keys:", ref.keys() if ref else "EMPTY")
print("Reference sample (first 200 chars of str):", str(ref)[:200])
