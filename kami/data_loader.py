import json
from typing import Iterator, Dict


def load_jsonl(path: str) -> Iterator[Dict]:
    """Yield JSON objects from a .jsonl file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
                continue
