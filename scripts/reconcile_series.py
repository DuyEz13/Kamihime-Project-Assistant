from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kami.paths import DATA_DIR  # noqa: E402
from kami.series import reconcile_series_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile registry-driven series metadata in local raw and "
            "translated data without crawling, translating, or rebuilding RAG."
        )
    )
    parser.add_argument(
        "--object-type",
        choices=("kamihime", "eidolon", "weapon"),
        default="kamihime",
    )
    args = parser.parse_args()
    result = reconcile_series_data(
        DATA_DIR,
        args.object_type,
        sync_translations=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
