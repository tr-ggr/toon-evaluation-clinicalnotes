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
    parser.add_argument("--model", choices=["deepseek-r1-turbo", "openai/gpt-4-turbo", "anthropic/claude-3.5-sonnet"], default="deepseek-r1-turbo")
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
    samples = list(iter_acn_hf(limit=args.limit))
    results = parse_dataset(samples, settings)
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    total_time = sum(r.elapsed_seconds for r in results)
    print(f"Completed parsing: {succeeded} succeeded, {failed} failed, total_time={total_time:.2f}s")


if __name__ == "__main__":
    main()
