from __future__ import annotations

from typing import Any, Dict

from toon_experiment.formats import json_format, yaml_format, toon_format


def convert_reference(ref_json: Dict[str, Any], target_format: str) -> str:
    """Convert reference dict to target format string."""
    if target_format == "json":
        return json_format.dumps(ref_json)
    if target_format == "yaml":
        return yaml_format.dumps(ref_json)
    if target_format == "toon":
        return toon_format.dumps(ref_json)
    raise ValueError(f"Unknown format: {target_format}")
