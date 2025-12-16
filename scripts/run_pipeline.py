#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from toon_experiment.config import Settings
from toon_experiment.io.datasets import iter_acn_hf
from toon_experiment.pipeline.run import parse_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parsing pipeline for ACN notes")
    parser.add_argument("--format", choices=["json", "yaml", "toon"], default="json")
    parser.add_argument("--model", choices=["gemini-2.5-pro"], default="gemini-2.5-pro")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        format=args.format,
        model=args.model,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        max_retries=args.max_retries or Settings().max_retries,
        temperature=args.temperature or Settings().temperature,
        top_p=args.top_p or Settings().top_p,
        seed=args.seed,
    )
    print("=" * 60)
    print("Pipeline Settings:")
    print(f"  Format: {settings.format}")
    print(f"  Model: {settings.model}")
    print(f"  Data Dir: {settings.data_dir}")
    print(f"  Outputs Dir: {settings.outputs_dir}")
    print(f"  Max Retries: {settings.max_retries}")
    print(f"  Temperature: {settings.temperature}")
    print(f"  Top P: {settings.top_p}")
    print(f"  Seed: {settings.seed}")
    print(f"  Limit: {args.limit}")
    print("=" * 60)
    samples = list(iter_acn_hf(limit=args.limit))
    results = parse_dataset(samples, settings)
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    total_time = sum(r.elapsed_seconds for r in results)
    print(f"Completed parsing: {succeeded} succeeded, {failed} failed, total_time={total_time:.2f}s")


if __name__ == "__main__":
    main()
