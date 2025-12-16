import json
from pathlib import Path

# Load sample
sample_path = Path("scripts/outputs/gemini-2.5-pro/json/sample_00000.json")
with open(sample_path) as f:
    data = json.load(f)

# Parse output
import sys
sys.path.insert(0, "/home/tr-ggr/NerdProjects/toon-experiment-v1/src")
from toon_experiment.eval.run_eval import _strip_markdown_code_block, _extract_prediction, _extract_ground_truth

pred_str = data["output_summary"]
pred_str = _strip_markdown_code_block(pred_str)
pred = json.loads(pred_str)

gt_str = data["ground_truth_summary"]
gt = json.loads(gt_str)

# Flatten
def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep).items())
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    items.extend(flatten(item, f"{new_key}[{i}]", sep).items())
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, str(v)))
    return dict(items)

pflat = flatten(pred)
rflat = flatten(gt)

print(f"Prediction flattened keys: {len(pflat)}")
print(f"Reference flattened keys: {len(rflat)}")
print()

print("Prediction keys (first 15):")
for k in list(pflat.keys())[:15]:
    print(f"  {k}")

print("\nReference keys (first 15):")
for k in list(rflat.keys())[:15]:
    print(f"  {k}")

print("\nCommon keys:")
common = set(pflat.keys()) & set(rflat.keys())
print(f"  {len(common)} common keys")
if common:
    for k in sorted(common)[:10]:
        print(f"    {k}: pred={repr(pflat[k][:30])} vs ref={repr(rflat[k][:30])}")
