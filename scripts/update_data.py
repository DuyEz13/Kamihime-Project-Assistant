"""Run the local data pipeline without starting the web application."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    from kami.local_data import LocalDataStore
    from kami.paths import OBJECT_TYPES, TRANSLATION_PROVIDERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-type", choices=OBJECT_TYPES, action="append", dest="object_types")
    parser.add_argument("--provider", choices=TRANSLATION_PROVIDERS, default="deepl")
    parser.add_argument("--mode", choices=("latest", "database", "translate"), default="latest")
    parser.add_argument("--scheduled", action="store_true", help="Apply unattended input guard")
    parser.add_argument("--restart", action="store_true", help="Discard this job's failed staging and check the source again")
    parser.add_argument("--skip-index", action="store_true", help="Publish wiki only; leave index pending")
    parser.add_argument("--status", action="store_true", help="Read saved pipeline state without network calls")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration only; no source/API calls or writes")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(LocalDataStore().status(), ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({"object_types": args.object_types or list(OBJECT_TYPES),
                          "provider": args.provider, "mode": args.mode,
                          "scheduled": args.scheduled, "build_index": not args.skip_index,
                          "index_device": "cuda"}, indent=2))
        return 0
    from kami.pipeline_runner import run_updates
    result = run_updates(args.object_types or OBJECT_TYPES, mode=args.mode, provider=args.provider,
                         trigger="scheduled" if args.scheduled else "manual", restart=args.restart,
                         build_index=not args.skip_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["state"] == "busy":
        return 0
    return 1 if result["state"] == "failed" or result.get("index_state") == "failed" else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
