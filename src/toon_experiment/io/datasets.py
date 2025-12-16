from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore


@dataclass
class ACNSample:
    full_note: str
    summary: dict
    source_path: Path


def load_acn_jsonl(path: Path, limit: Optional[int] = None) -> List[ACNSample]:
    records: List[ACNSample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            obj = json.loads(line)
            full_note = obj.get("full_note") or obj.get("note") or ""
            summary = obj.get("summary") or {}
            records.append(ACNSample(full_note=full_note, summary=summary, source_path=path))
    return records


def iter_acn_dir(directory: Path, limit: Optional[int] = None) -> Iterator[ACNSample]:
    count = 0
    for path in sorted(directory.glob("*.jsonl")):
        for rec in load_acn_jsonl(path):
            yield rec
            count += 1
            if limit is not None and count >= limit:
                return


def iter_acn_hf(dataset_name: str = "AGBonnet/augmented-clinical-notes", split: str = "train", limit: Optional[int] = None) -> Iterator[ACNSample]:
    if load_dataset is None:
        raise ImportError("datasets is not installed; run `pip install datasets`")
    ds = load_dataset(dataset_name, split=split)  # type: ignore[operator]
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        full_note = row.get("full_note") or row.get("note") or ""
        summary = row.get("summary") or {}
        yield ACNSample(full_note=full_note, summary=summary, source_path=Path(dataset_name))
