from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


class FormatError(Exception):
    pass


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]


Serializer = Callable[[Dict[str, Any]], str]
Parser = Callable[[str], Dict[str, Any]]
Validator = Callable[[Dict[str, Any]], ValidationResult]
