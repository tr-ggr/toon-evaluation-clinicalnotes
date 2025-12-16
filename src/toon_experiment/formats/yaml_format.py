from __future__ import annotations

from typing import Any, Dict

import yaml

from toon_experiment.formats.base import ValidationResult
from toon_experiment.schemas.summary import Summary, summary_template

YAML_TEMPLATE: Dict[str, Any] = summary_template()


def dumps(obj: Dict[str, Any]) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def loads(text: str) -> Dict[str, Any]:
    return yaml.safe_load(text) or {}


def validate(obj: Dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    try:
        Summary.model_validate(obj)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(str(exc))
    return ValidationResult(valid=not errors, errors=errors)
