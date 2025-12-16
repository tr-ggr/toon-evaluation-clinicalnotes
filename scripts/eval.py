#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

from toon_experiment.config import Settings
from toon_experiment.eval.run_eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate parsed outputs against ACN summaries")
    parser.add_argument("--format", choices=["json", "yaml", "toon", "all"], default="json")
    parser.add_argument("--model", choices=["gemini-2.5-pro"], default="gemini-2.5-pro")
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).parent / "outputs")
    return parser.parse_args()


def print_single_format(results: Dict[str, Any], fmt: str, model: str) -> None:
    """Print evaluation results for a single format."""
    print("=" * 80)
    print(f"EVALUATION RESULTS: {model} - {fmt.upper()}")
    print("=" * 80)
    
    # Field-level metrics
    print("\n--- FIELD-LEVEL METRICS ---")
    print(f"Precision:  {results['field_precision']:.3f}")
    print(f"Recall:     {results['field_recall']:.3f}")
    print(f"F1:         {results['field_f1']:.3f}")
    print(f"BERTScore:  {results['bertscore_avg']:.3f}")
    
    # Entity-level metrics
    print("\n--- ENTITY-LEVEL METRICS (Arrays) ---")
    entity_metrics = results.get("entity_metrics", {})
    if entity_metrics:
        for field, metrics in sorted(entity_metrics.items()):
            print(f"{field:25s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
    else:
        print("No entity metrics available")
    
    # Coverage metrics
    print("\n--- SCHEMA COVERAGE (Field Population) ---")
    coverage = results.get("coverage", {})
    if coverage:
        total_coverage = sum(coverage.values()) / len(coverage) if coverage else 0.0
        for section, rate in sorted(coverage.items()):
            bar = "█" * int(rate * 20)
            print(f"{section:30s}  {rate:5.1%}  {bar}")
        print(f"{'AVERAGE':30s}  {total_coverage:5.1%}")
    else:
        print("No coverage metrics available")
    
    # Format comparison metrics
    print("\n--- FORMAT COMPARISON METRICS ---")
    format_metrics = results.get("format_metrics", {})
    print(f"Success Rate:    {format_metrics.get('success_rate', 0.0):5.1%}")
    print(f"Avg Retries:     {format_metrics.get('avg_retries', 0.0):.1f}")
    print(f"Avg Time (s):    {format_metrics.get('avg_time', 0.0):.1f}")
    print(f"Total Samples:   {format_metrics.get('total_samples', 0)}")
    print(f"Failed Samples:  {format_metrics.get('failed_samples', 0)}")
    
    print("=" * 80)


def print_comparison(results_by_format: Dict[str, Any]) -> None:
    """Print a comparison of metrics across all formats."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FORMAT COMPARISON")
    print("=" * 80)
    
    formats = sorted(results_by_format.keys())
    if not formats:
        print("No results to compare")
        return
    
    # Field-level metrics comparison
    print("\n--- FIELD-LEVEL METRICS COMPARISON ---")
    print(f"{'Format':<12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'BERTScore':>10}")
    print("-" * 60)
    for fmt in formats:
        results = results_by_format[fmt]
        print(f"{fmt:<12}  {results['field_precision']:>10.3f}  {results['field_recall']:>10.3f}  {results['field_f1']:>10.3f}  {results['bertscore_avg']:>10.3f}")
    
    # Entity-level metrics comparison
    print("\n--- ENTITY-LEVEL METRICS BY FORMAT ---")
    all_fields = set()
    for results in results_by_format.values():
        all_fields.update(results.get("entity_metrics", {}).keys())
    
    for field in sorted(all_fields):
        print(f"\n{field.upper()}:")
        print(f"{'Format':<12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
        print("-" * 45)
        for fmt in formats:
            metrics = results_by_format[fmt].get("entity_metrics", {}).get(field)
            if metrics:
                print(f"{fmt:<12}  {metrics['precision']:>10.3f}  {metrics['recall']:>10.3f}  {metrics['f1']:>10.3f}")
            else:
                print(f"{fmt:<12}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
    
    # Coverage comparison
    print("\n--- SCHEMA COVERAGE COMPARISON ---")
    all_sections = set()
    for results in results_by_format.values():
        all_sections.update(results.get("coverage", {}).keys())
    
    print(f"{'Section':<30}  " + "  ".join(f"{fmt:>10}" for fmt in formats))
    print("-" * (30 + 2 + 12 * len(formats)))
    for section in sorted(all_sections):
        vals = []
        for fmt in formats:
            coverage = results_by_format[fmt].get("coverage", {}).get(section, 0.0)
            vals.append(f"{coverage:>9.1%}")
        print(f"{section:<30}  " + "  ".join(vals))
    
    # Format comparison metrics
    print("\n--- FORMAT RELIABILITY METRICS ---")
    print(f"{'Format':<12}  {'Success %':>12}  {'Avg Retries':>13}  {'Avg Time(s)':>12}  {'Failed':>7}")
    print("-" * 65)
    for fmt in formats:
        fm = results_by_format[fmt].get("format_metrics", {})
        print(f"{fmt:<12}  {fm.get('success_rate', 0.0):>11.1%}  {fm.get('avg_retries', 0.0):>13.1f}  {fm.get('avg_time', 0.0):>12.1f}  {fm.get('failed_samples', 0):>7}")
    
    print("\n" + "=" * 80)


def main() -> None:
    args = parse_args()
    
    if args.format == "all":
        # Evaluate all formats
        formats = ["json", "yaml", "toon"]
        results_by_format = {}
        for fmt in formats:
            settings = Settings(format=fmt, model=args.model, outputs_dir=args.outputs_dir)
            out_dir = settings.outputs_dir / settings.model / settings.format
            results_by_format[fmt] = evaluate(out_dir)
        print_comparison(results_by_format)
    else:
        # Evaluate single format
        settings = Settings(format=args.format, model=args.model, outputs_dir=args.outputs_dir)
        out_dir = settings.outputs_dir / settings.model / settings.format
        results = evaluate(out_dir)
        print_single_format(results, args.format, settings.model)


if __name__ == "__main__":
    main()

