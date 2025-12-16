#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from toon_experiment.config import Settings
from toon_experiment.eval.run_eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate parsed outputs against ACN summaries")
    parser.add_argument("--format", choices=["json", "yaml", "toon"], default="json")
    parser.add_argument("--model", choices=["gemini-2.5-pro"], default="gemini-2.5-pro")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(format=args.format, model=args.model, outputs_dir=args.outputs_dir)
    out_dir = settings.outputs_dir / settings.model / settings.format
    precision, recall, f1, bert = evaluate(out_dir)
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} BERTScore={bert:.3f}")


if __name__ == "__main__":
    main()
