from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from toon_experiment.config import OutputFormat, Settings
from toon_experiment.formats import json_format, yaml_format, toon_format
from toon_experiment.io.datasets import ACNSample
from toon_experiment.pipeline.models import get_chat_model
from toon_experiment.prompts.templates import build_prompt

FORMAT_MODULES = {
    "json": json_format,
    "yaml": yaml_format,
    "toon": toon_format,
}


@dataclass
class ParseResult:
    sample_idx: int
    attempts: int
    success: bool
    output_path: Path
    parsed_path: Path
    errors: List[str]
    elapsed_seconds: float


def _prepare_output_dirs(outputs_dir: Path, model: str, fmt: str) -> Path:
    base = outputs_dir / model / fmt
    base.mkdir(parents=True, exist_ok=True)
    return base


def _parse_text(raw: str, fmt: OutputFormat) -> Dict:
    """Parse LLM output, stripping markdown code blocks if present."""
    # Strip markdown code blocks (```json ... ``` or ```yaml ... ``` or ```toon ... ```)
    text = raw.strip()
    if text.startswith("```"):
        # Find first newline after opening ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing ```
        if text.endswith("```"):
            text = text[:-3].rstrip()
    
    if fmt == "json":
        return json_format.loads(text)
    if fmt == "yaml":
        return yaml_format.loads(text)
    if fmt == "toon":
        return toon_format.loads(text)
    raise ValueError(f"Unknown format: {fmt}")


def parse_sample(sample: ACNSample, sample_idx: int, settings: Settings) -> ParseResult:
    fmt = settings.format
    fmt_module = FORMAT_MODULES[fmt]
    llm = get_chat_model(settings.model, settings.temperature, settings.top_p, settings.seed)
    prompt = build_prompt(fmt).format(clinical_note=sample.full_note)

    out_dir = _prepare_output_dirs(settings.outputs_dir, settings.model, fmt)
    raw_path = out_dir / f"sample_{sample_idx:05d}.txt"
    parsed_path = out_dir / f"sample_{sample_idx:05d}.json"

    print(f"[Sample {sample_idx:05d}] Starting parse (format={fmt}, model={settings.model})")
    errors: List[str] = []
    start = time.perf_counter()
    attempts = 0
    success = False
    parsed_obj: Optional[Dict] = None
    last_content = ""
    last_error = ""
    for attempt in range(settings.max_retries + 1):
        attempts += 1
        attempt_start = time.perf_counter()
        
        # On retries, append error context to prompt
        current_prompt = prompt
        if attempt > 0 and last_error:
            current_prompt = (
                f"{prompt}\n\n"
                f"PREVIOUS ATTEMPT FAILED WITH ERRORS:\n{last_error}\n\n"
                f"PREVIOUS OUTPUT:\n{last_content}\n\n"
                f"Please correct the errors and provide a valid {fmt.upper()} response."
            )
        
        try:
            response = llm.invoke(current_prompt)
            content = getattr(response, "content", None)
            if content is None:
                content = str(response)
            last_content = content
            
            # Debug: show first 200 chars of response on failures
            if attempts > 1:
                preview = content[:200].replace('\n', '\\n')
                print(f"[Sample {sample_idx:05d}] Response preview: {preview}...")
            
            candidate = _parse_text(content, fmt)
            validation = fmt_module.validate(candidate)
            attempt_time = time.perf_counter() - attempt_start
            if validation.valid:
                parsed_obj = candidate
                success = True
                with raw_path.open("w", encoding="utf-8") as f:
                    f.write(content)
                output_obj = dict(candidate)
                try:
                    output_obj["ground_truth_summary"] = fmt_module.dumps(sample.summary)
                except Exception:
                    output_obj["ground_truth_summary"] = sample.summary
                with parsed_path.open("w", encoding="utf-8") as f:
                    json.dump(output_obj, f, ensure_ascii=False, indent=2)
                print(f"[Sample {sample_idx:05d}] SUCCESS on attempt {attempts} ({attempt_time:.2f}s)")
                break
            else:
                error_msg = "; ".join(validation.errors)
                errors.append(error_msg)
                last_error = error_msg
                print(
                    f"[Sample {sample_idx:05d}] Validation failed on attempt {attempts} ({attempt_time:.2f}s): {error_msg}"
                )
        except Exception as exc:  # pragma: no cover - defensive for malformed outputs
            attempt_time = time.perf_counter() - attempt_start
            error_msg = str(exc)
            errors.append(error_msg)
            last_error = error_msg
            print(f"[Sample {sample_idx:05d}] Exception on attempt {attempts} ({attempt_time:.2f}s): {error_msg}")
        if attempt < settings.max_retries:
            continue
    elapsed = time.perf_counter() - start
    if not success and parsed_obj is None:
        # still write raw last response for debugging
        with raw_path.open("w", encoding="utf-8") as f:
            f.write(last_content)
        print(
            f"[Sample {sample_idx:05d}] FAILED after {attempts} attempts ({elapsed:.2f}s total). Errors: {errors}"
        )
    return ParseResult(
        sample_idx=sample_idx,
        attempts=attempts,
        success=success,
        output_path=raw_path,
        parsed_path=parsed_path,
        errors=errors,
        elapsed_seconds=elapsed,
    )


def _write_run_summary(results: List[ParseResult], output_dir: Path) -> None:
    """Write run summary to file."""
    summary_path = output_dir / "run_summary.txt"
    
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    total_attempts = sum(r.attempts for r in results)
    total_retries = total_attempts - len(results)
    avg_time = sum(r.elapsed_seconds for r in results) / len(results) if results else 0.0
    
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RUN SUMMARY - PER SAMPLE DETAILS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Sample':<10} {'Status':<10} {'Attempts':<10} {'Retries':<10} {'Time (s)':<12}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            status = "SUCCESS" if r.success else "FAILED"
            retries = r.attempts - 1
            f.write(f"{r.sample_idx:05d}      {status:<10} {r.attempts:<10} {retries:<10} {r.elapsed_seconds:<12.2f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<10} {succeeded}/{len(results):<9} {total_attempts:<10} {total_retries:<10} {sum(r.elapsed_seconds for r in results):<12.2f}\n")
        f.write(f"{'AVERAGE':<10} {'':10} {total_attempts/len(results):<10.2f} {total_retries/len(results):<10.2f} {avg_time:<12.2f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nRun summary saved to: {summary_path}")


def parse_dataset(samples: List[ACNSample], settings: Settings) -> List[ParseResult]:
    results: List[ParseResult] = []
    print(f"Starting parsing of {len(samples)} samples (format={settings.format}, model={settings.model})")
    for idx, sample in enumerate(samples):
        res = parse_sample(sample, idx, settings)
        results.append(res)
    
    # Summary statistics
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    total_attempts = sum(r.attempts for r in results)
    avg_time = sum(r.elapsed_seconds for r in results) / len(results) if results else 0.0
    total_retries = total_attempts - len(results)  # retries = total attempts - samples
    
    print(f"Parsing completed: {succeeded}/{len(results)} succeeded, {failed} failed")
    print(f"Total attempts: {total_attempts}, Total retries: {total_retries}")
    print(f"Average time per sample: {avg_time:.2f}s")
    
    # Detailed per-sample summary
    print("\n" + "=" * 80)
    print("RUN SUMMARY - PER SAMPLE DETAILS")
    print("=" * 80)
    print(f"{'Sample':<10} {'Status':<10} {'Attempts':<10} {'Retries':<10} {'Time (s)':<12}")
    print("-" * 80)
    for r in results:
        status = "SUCCESS" if r.success else "FAILED"
        retries = r.attempts - 1
        print(f"{r.sample_idx:05d}      {status:<10} {r.attempts:<10} {retries:<10} {r.elapsed_seconds:<12.2f}")
    print("-" * 80)
    print(f"{'TOTAL':<10} {succeeded}/{len(results):<9} {total_attempts:<10} {total_retries:<10} {sum(r.elapsed_seconds for r in results):<12.2f}")
    print(f"{'AVERAGE':<10} {'':10} {total_attempts/len(results):<10.2f} {total_retries/len(results):<10.2f} {avg_time:<12.2f}")
    print("=" * 80)
    
    # Write summary to file
    output_dir = _prepare_output_dirs(settings.outputs_dir, settings.model, settings.format)
    _write_run_summary(results, output_dir)
    
    return results
