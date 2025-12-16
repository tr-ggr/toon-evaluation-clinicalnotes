# TOON Experiment V1

LangChain-based pipeline for structured clinical information extraction from unstructured clinical notes into JSON, YAML, and Token-Oriented Object Notation (TOON), with validation/re-parse loops and evaluation (precision/recall/F1, BERTScore).

## Status

MVP implementation with parsing, validation/retry loop, and evaluation. Configure model endpoints and TOON CLI path before running.

## Quickstart

1. Install deps (recommend venv): `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your Google Gemini API key (`GOOGLE_API_KEY=...`). `python-dotenv` auto-loads `.env` via `config.py`.
3. Provide ACN dataset `full_note` + `summary` JSON lines under `data/*.jsonl`, or use Hugging Face: `iter_acn_hf()` loads `AGBonnet/augmented-clinical-notes`.
4. Run parsing for one format: `python scripts/run_pipeline.py --format json --model gemini-2.5-pro-exp --limit 10`
5. Run evaluation: `python scripts/eval.py --format json --model gemini-2.5-pro-exp`

## Layout

- `src/toon_experiment/`: core package
  - `schemas/summary.py`: canonical summary schema and defaults
  - `formats/`: converters/validators for JSON/YAML/TOON (TOON uses external CLI)
  - `prompts/`: prompt templates per format
  - `pipeline/`: parsing pipeline, retry logic, and model selection
  - `eval/`: evaluation metrics (field P/R/F1, BERTScore) and runners
- `scripts/`: entrypoints for parsing and evaluation

## Notes

- TOON encoding/decoding uses the `python-toon` library (included in `requirements.txt`).
- Long-context models may require endpoint-specific settings; see `pipeline/models.py`.
- Evaluation expects parsed outputs under `outputs/{model}/{format}` matching `scripts/run_pipeline.py` naming.
