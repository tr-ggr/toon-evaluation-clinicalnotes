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


def _parse_text(raw: str, fmt: OutputFormat, settings: Settings) -> Dict:
    if fmt == "json":
        return json_format.loads(raw)
    if fmt == "yaml":
        return yaml_format.loads(raw)
    if fmt == "toon":
        return toon_format.toon_to_json(raw)
    raise ValueError(f"Unknown format: {fmt}")


def parse_sample(sample: ACNSample, sample_idx: int, settings: Settings) -> ParseResult:
    fmt = settings.format
    fmt_module = FORMAT_MODULES[fmt]
    llm = get_chat_model(settings.model, settings.temperature, settings.top_p, settings.seed)
    prompt = build_prompt(fmt).format(clinical_note=sample.full_note)

    out_dir = _prepare_output_dirs(settings.outputs_dir, settings.model, fmt)
    raw_path = out_dir / f"sample_{sample_idx:05d}.txt"
    parsed_path = out_dir / f"sample_{sample_idx:05d}.json"

    errors: List[str] = []
    start = time.perf_counter()
    attempts = 0
    success = False
    parsed_obj: Optional[Dict] = None
    last_content = ""
    for attempt in range(settings.max_retries + 1):
        attempts += 1
        response = llm.invoke(prompt)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        last_content = content
        try:
            candidate = _parse_text(content, fmt, settings)
            validation = fmt_module.validate(candidate)
            if validation.valid:
                parsed_obj = candidate
                success = True
                with raw_path.open("w", encoding="utf-8") as f:
                    f.write(content)
                with parsed_path.open("w", encoding="utf-8") as f:
                    json.dump(candidate, f, ensure_ascii=False, indent=2)
                break
            else:
                errors.append("; ".join(validation.errors))
        except Exception as exc:  # pragma: no cover - defensive for malformed outputs
            errors.append(str(exc))
        if attempt < settings.max_retries:
            continue
    elapsed = time.perf_counter() - start
    if not success and parsed_obj is None:
        # still write raw last response for debugging
        with raw_path.open("w", encoding="utf-8") as f:
            f.write(last_content)
    return ParseResult(
        sample_idx=sample_idx,
        attempts=attempts,
        success=success,
        output_path=raw_path,
        parsed_path=parsed_path,
        errors=errors,
        elapsed_seconds=elapsed,
    )


def parse_dataset(samples: List[ACNSample], settings: Settings) -> List[ParseResult]:
    results: List[ParseResult] = []
    for idx, sample in enumerate(samples):
        res = parse_sample(sample, idx, settings)
        results.append(res)
    return results
