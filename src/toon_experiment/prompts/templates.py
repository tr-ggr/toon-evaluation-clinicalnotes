from __future__ import annotations

from textwrap import dedent
from typing import Dict

from toon_experiment.formats import json_format, yaml_format, toon_format
from toon_experiment.schemas.summary import summary_template

PROMPT_PREAMBLE = """
You are a clinical information extraction system. Given a clinical note, produce a structured summary strictly following the provided target format and schema. Do not include any explanatory text or prose outside the serialized structure. If information is missing, leave the field null/empty but preserve the field.
"""


def format_template(target_format: str) -> str:
    tmpl_dict: Dict[str, object] = summary_template()
    if target_format == "json":
        return json_format.dumps(tmpl_dict)
    if target_format == "yaml":
        return yaml_format.dumps(tmpl_dict)
    if target_format == "toon":
        # For TOON we show JSON template to reduce prompt size; generator will output TOON.
        return json_format.dumps(tmpl_dict)
    raise ValueError(f"Unknown format: {target_format}")


def build_prompt(target_format: str) -> str:
    tmpl = format_template(target_format)
    format_label = target_format.upper()
    return dedent(
        f"""
        {PROMPT_PREAMBLE.strip()}

        TARGET FORMAT: {format_label}
        SCHEMA TEMPLATE:
        ```
        {tmpl}
        ```

        RULES:
        - Output must be valid {format_label} and parse without errors.
        - Preserve all fields from the template; do not drop keys.
        - Use empty strings or nulls if information is unavailable.
        - Do not add commentary or extra keys.

        INPUT CLINICAL NOTE:
        ```
        {{clinical_note}}
        ```

        RESPONSE: Only the {format_label} structure.
        """
    ).strip()
