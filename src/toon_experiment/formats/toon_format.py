from __future__ import annotations

from typing import Any, Dict

try:
    from toon import decode, encode, DecodeOptions, EncodeOptions  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    decode = None  # type: ignore
    encode = None  # type: ignore
    DecodeOptions = None  # type: ignore
    EncodeOptions = None  # type: ignore

from toon_experiment.formats.base import FormatError, ValidationResult
from toon_experiment.schemas.summary import Summary, summary_template

TOON_TEMPLATE: Dict[str, Any] = summary_template()


def dumps(obj: Dict[str, Any]) -> str:
    """Encode dict to TOON format string."""
    if encode is None:
        raise FormatError("python-toon is not installed; run `pip install python-toon`")
    try:
        return encode(obj, EncodeOptions(indent=2))  # type: ignore[operator]
    except Exception as exc:  # pragma: no cover - defensive
        raise FormatError(f"TOON encoding failed: {exc}")


def loads(text: str) -> Dict[str, Any]:
    """Decode TOON format string to dict."""
    if decode is None:
        raise FormatError("python-toon is not installed; run `pip install python-toon`")
    try:
        return decode(text, DecodeOptions(indent=2, strict=True))  # type: ignore[operator]
    except Exception as exc:  # pragma: no cover - defensive
        raise FormatError(f"TOON decoding failed: {exc}")


def validate(obj: Dict[str, Any]) -> ValidationResult:
    """Validate dict against Summary schema."""
    errors: list[str] = []
    try:
        Summary.model_validate(obj)
    except Exception as exc:
        errors.append(str(exc))
    return ValidationResult(valid=not errors, errors=errors)
