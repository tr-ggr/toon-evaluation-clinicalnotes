#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

from toon_experiment.config import Settings
from toon_experiment.eval.run_eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate parsed outputs against ACN summaries")
    parser.add_argument("--format", choices=["json", "yaml", "toon", "all"], default="json")
    parser.add_argument("--model", choices=["gemini-2.5-pro"], default="gemini-2.5-pro")
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--output-file", type=Path, default=Path(__file__).parent / "results.md")
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
    
    # Compression metrics
    print("\n--- TOKEN COMPRESSION ---")
    compression = results.get("compression", {})
    print(f"Avg Input Chars/Sample:     {compression.get('average_input_chars', 0):,.0f}")
    print(f"Avg Output Chars/Sample:    {compression.get('average_output_chars', 0):,.0f}")
    print(f"Average Compression:        {compression.get('average_compression', 0.0):.1%}")
    
    if compression.get("per_sample"):
        print("\nPer-Sample Compression:")
        for sample_data in compression["per_sample"]:
            print(f"  Sample {sample_data['sample']:05d}: {sample_data['compression']:.1%}")
    
    print("=" * 80)


def format_single_markdown(results: Dict[str, Any], fmt: str, model: str) -> List[str]:
    """Format evaluation results for a single format as markdown."""
    lines = [
        f"# {model.upper()} - {fmt.upper()} Evaluation Results",
        "",
        "## Field-Level Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Precision | {results['field_precision']:.3f} |",
        f"| Recall | {results['field_recall']:.3f} |",
        f"| F1 Score | {results['field_f1']:.3f} |",
        f"| BERTScore | {results['bertscore_avg']:.3f} |",
        "",
    ]
    
    # Entity-level metrics
    entity_metrics = results.get("entity_metrics", {})
    if entity_metrics:
        lines.extend([
            "## Entity-Level Metrics (Arrays)",
            "",
            "| Field | Precision | Recall | F1 |",
            "|-------|-----------|--------|-----|",
        ])
        for field, metrics in sorted(entity_metrics.items()):
            lines.append(f"| {field} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
        lines.append("")
    
    # Coverage metrics
    coverage = results.get("coverage", {})
    if coverage:
        total_coverage = sum(coverage.values()) / len(coverage) if coverage else 0.0
        lines.extend([
            "## Schema Coverage (Field Population)",
            "",
            "| Section | Coverage |",
            "|---------|----------|",
        ])
        for section, rate in sorted(coverage.items()):
            lines.append(f"| {section} | {rate:.1%} |")
        lines.append(f"| **AVERAGE** | **{total_coverage:.1%}** |")
        lines.append("")
    
    # Format metrics
    format_metrics = results.get("format_metrics", {})
    lines.extend([
        "## Format Reliability Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Success Rate | {format_metrics.get('success_rate', 0.0):.1%} |",
        f"| Avg Retries | {format_metrics.get('avg_retries', 0.0):.1f} |",
        f"| Avg Time (s) | {format_metrics.get('avg_time', 0.0):.1f} |",
        f"| Total Samples | {format_metrics.get('total_samples', 0)} |",
        f"| Failed Samples | {format_metrics.get('failed_samples', 0)} |",
        "",
    ])
    
    # Compression metrics
    compression = results.get("compression", {})
    lines.extend([
        "## Token Compression",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Input Characters | {compression.get('total_input_chars', 0):,} |",
        f"| Output Characters | {compression.get('total_output_chars', 0):,} |",
        f"| Average Compression | {compression.get('average_compression', 0.0):.1%} |",
        "",
    ])
    
    if compression.get("per_sample"):
        lines.extend([
            "### Per-Sample Compression",
            "",
            "| Sample | Compression |",
            "|--------|------------|",
        ])
        for sample_data in compression["per_sample"]:
            lines.append(f"| {sample_data['sample']:05d} | {sample_data['compression']:.1%} |")
        lines.append("")
    
    return lines


def format_comparison_markdown(results_by_format: Dict[str, Any]) -> List[str]:
    """Format comparison of metrics across all formats as markdown."""
    lines = [
        "# Comprehensive Format Comparison",
        "",
    ]
    
    formats = sorted(results_by_format.keys())
    if not formats:
        return lines + ["No results to compare"]
    
    # Field-level metrics comparison
    lines.extend([
        "## Field-Level Metrics Comparison",
        "",
        "| Format | Precision | Recall | F1 | BERTScore |",
        "|--------|-----------|--------|-----|-----------|",
    ])
    for fmt in formats:
        results = results_by_format[fmt]
        lines.append(f"| {fmt} | {results['field_precision']:.3f} | {results['field_recall']:.3f} | {results['field_f1']:.3f} | {results['bertscore_avg']:.3f} |")
    lines.append("")
    
    # Entity-level metrics comparison
    all_fields = set()
    for results in results_by_format.values():
        all_fields.update(results.get("entity_metrics", {}).keys())
    
    if all_fields:
        lines.extend(["## Entity-Level Metrics by Format", ""])
        for field in sorted(all_fields):
            lines.extend([
                f"### {field.replace('_', ' ').title()}",
                "",
                "| Format | Precision | Recall | F1 |",
                "|--------|-----------|--------|-----|",
            ])
            for fmt in formats:
                metrics = results_by_format[fmt].get("entity_metrics", {}).get(field)
                if metrics:
                    lines.append(f"| {fmt} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
                else:
                    lines.append(f"| {fmt} | N/A | N/A | N/A |")
            lines.append("")
    
    # Coverage comparison
    all_sections = set()
    for results in results_by_format.values():
        all_sections.update(results.get("coverage", {}).keys())
    
    if all_sections:
        lines.extend([
            "## Schema Coverage Comparison",
            "",
            "| Section | " + " | ".join(formats) + " |",
            "|" + "|".join(["---------"] * (len(formats) + 1)) + "|",
        ])
        for section in sorted(all_sections):
            row = f"| {section} |"
            for fmt in formats:
                coverage = results_by_format[fmt].get("coverage", {}).get(section, 0.0)
                row += f" {coverage:.1%} |"
            lines.append(row)
        lines.append("")
    
    # Format reliability metrics
    lines.extend([
        "## Format Reliability Metrics",
        "",
        "| Format | Success % | Avg Retries | Avg Time(s) | Failed |",
        "|--------|-----------|-------------|------------|--------|",
    ])
    for fmt in formats:
        fm = results_by_format[fmt].get("format_metrics", {})
        lines.append(f"| {fmt} | {fm.get('success_rate', 0.0):.1%} | {fm.get('avg_retries', 0.0):.1f} | {fm.get('avg_time', 0.0):.1f} | {fm.get('failed_samples', 0)} |")
    lines.append("")
    
    # Compression metrics
    lines.extend([
        "## Token Compression Comparison",
        "",
        "| Format | Input Chars | Output Chars | Average Compression |",
        "|--------|-------------|--------------|-------------------|",
    ])
    for fmt in formats:
        comp = results_by_format[fmt].get("compression", {})
        lines.append(f"| {fmt} | {comp.get('total_input_chars', 0):,} | {comp.get('total_output_chars', 0):,} | {comp.get('average_compression', 0.0):.1%} |")
    lines.append("")
    
    return lines


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
    
    # Compression metrics
    print("\n--- TOKEN COMPRESSION COMPARISON ---")
    print(f"{'Format':<12}  {'Samples':>8}  {'Avg Input/Sample':>16}  {'Avg Output/Sample':>17}  {'Avg Compression':>15}")
    print("-" * 80)
    for fmt in formats:
        comp = results_by_format[fmt].get("compression", {})
        num_samples = len(comp.get("per_sample", []))
        print(f"{fmt:<12}  {num_samples:>8}  {comp.get('average_input_chars', 0):>16,.0f}  {comp.get('average_output_chars', 0):>17,.0f}  {comp.get('average_compression', 0.0):>14.1%}")
    
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
        
        # Write to markdown file
        markdown_lines = format_comparison_markdown(results_by_format)
        with open(args.output_file, "w") as f:
            f.write("\n".join(markdown_lines))
        print(f"\nResults written to: {args.output_file}")
    else:
        # Evaluate single format
        settings = Settings(format=args.format, model=args.model, outputs_dir=args.outputs_dir)
        out_dir = settings.outputs_dir / settings.model / settings.format
        results = evaluate(out_dir)
        print_single_format(results, args.format, settings.model)
        
        # Write to markdown file
        markdown_lines = format_single_markdown(results, args.format, settings.model)
        with open(args.output_file, "w") as f:
            f.write("\n".join(markdown_lines))
        print(f"\nResults written to: {args.output_file}")


if __name__ == "__main__":
    main()

