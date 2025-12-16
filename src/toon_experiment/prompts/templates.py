from __future__ import annotations

from textwrap import dedent
from typing import Dict

from toon_experiment.formats import json_format, yaml_format, toon_format
from toon_experiment.schemas.summary import summary_template

PROMPT_PREAMBLE = """
You are a clinical information extraction system. Given a clinical note, produce a structured summary strictly following the provided target format and schema. Do not include any explanatory text or prose outside the serialized structure. If information is missing, leave the field null/empty but preserve the field.
"""

TOON_FORMAT_INSTRUCTIONS = """
Respond using TOON format (Token-Oriented Object Notation). Use key: value syntax, indentation for nesting, and tabular format [N,]{{fields}}: for uniform arrays. Array lengths are marked with [#N]. Ensure your response matches these counts.
"""


def format_template(target_format: str) -> str:
    tmpl_dict: Dict[str, object] = summary_template()
    if target_format == "json":
        return json_format.dumps(tmpl_dict)
    if target_format == "yaml":
        return yaml_format.dumps(tmpl_dict)
    if target_format == "toon":
        return toon_format.dumps(tmpl_dict)
    raise ValueError(f"Unknown format: {target_format}")


def build_prompt(target_format: str) -> str:
    tmpl = format_template(target_format)
    format_label = target_format.upper()
    # Escape braces in template to avoid format string conflicts
    tmpl_escaped = tmpl.replace("{", "{{").replace("}", "}}")
    
    # Add TOON-specific instructions
    format_instructions = ""
    if target_format == "toon":
        format_instructions = f"\n\n{TOON_FORMAT_INSTRUCTIONS.strip()}"
    
    prompt_template = dedent(
        f"""
        {PROMPT_PREAMBLE.strip()}{format_instructions}

        TARGET FORMAT: {format_label}
        SCHEMA TEMPLATE:
        ```{target_format}
        {tmpl_escaped}
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

        RESPONSE: Only the {format_label} structure wrapped in a ```{target_format} code block.
        """
    ).strip()
    print(f"Built prompt template:\n{prompt_template}\n")
    return prompt_template
