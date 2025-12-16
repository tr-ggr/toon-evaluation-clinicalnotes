from __future__ import annotations

import json as _json
from typing import Any, Dict, Optional

try:
    from toon import decode, DecodeOptions, encode, EncodeOptions  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    decode = None  # type: ignore
    encode = None  # type: ignore
    DecodeOptions = None  # type: ignore
    EncodeOptions = None  # type: ignore

from toon_experiment.formats.base import FormatError, ValidationResult
from toon_experiment.schemas.summary import Summary, summary_template

TOON_TEMPLATE: Dict[str, Any] = summary_template()


def json_to_toon(obj: Dict[str, Any]) -> str:
    if encode is None:
        raise FormatError("python-toon is not installed; run `pip install python-toon`")
    try:
        return encode(obj, EncodeOptions(indent=2))  # type: ignore[operator]
    except Exception as exc:  # pragma: no cover - defensive
        raise FormatError(f"TOON encoding failed: {exc}")


def toon_to_json(text: str) -> Dict[str, Any]:
    if decode is None:
        raise FormatError("python-toon is not installed; run `pip install python-toon`")
    try:
        return decode(text, DecodeOptions(indent=2, strict=True))  # type: ignore[operator]
    except Exception as exc:  # pragma: no cover - defensive
        raise FormatError(f"TOON decoding failed: {exc}")


def validate(obj: Dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    try:
        Summary.model_validate(obj)
    except Exception as exc:
        errors.append(str(exc))
    return ValidationResult(valid=not errors, errors=errors)
