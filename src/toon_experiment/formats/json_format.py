from __future__ import annotations

import json
from typing import Any, Dict

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

from toon_experiment.formats.base import ValidationResult
from toon_experiment.schemas.summary import Summary, summary_template

JSON_TEMPLATE: Dict[str, Any] = summary_template()


def dumps(obj: Dict[str, Any]) -> str:
    if orjson:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")
    return json.dumps(obj, indent=2, ensure_ascii=False)


def loads(text: str) -> Dict[str, Any]:
    if orjson:
        return orjson.loads(text)
    return json.loads(text)


def validate(obj: Dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    try:
        Summary.model_validate(obj)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(str(exc))
    return ValidationResult(valid=not errors, errors=errors)
