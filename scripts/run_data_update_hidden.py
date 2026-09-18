"""Run the scheduled local data update without creating a console window."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
PROVIDERS = ("deepl", "google", "qwen")


def run_hidden_update(
    project_root: Path,
    provider: str,
    *,
    process_runner: Callable = subprocess.run,
) -> int:
    """Run the normal worker with redirected output and no Windows console."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown translation provider: {provider}")

    project_root = Path(project_root).resolve()
    python = project_root / ".venv" / "Scripts" / "python.exe"
    worker = project_root / "scripts" / "update_data.py"
    if not python.is_file():
        raise FileNotFoundError("Run uv sync before starting automatic updates.")
    if not worker.is_file():
        raise FileNotFoundError("The scheduled update worker is missing.")

    log_path = project_root / "kami" / "data" / ".pipeline" / "scheduler.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(worker),
        "--object-type", "kamihime",
        "--object-type", "eidolon",
        "--object-type", "weapon",
        "--provider", provider,
        "--scheduled",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = process_runner(
            command,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default="deepl")
    args = parser.parse_args(argv)
    try:
        return run_hidden_update(ROOT_DIR, args.provider)
    except Exception as exc:
        log_path = ROOT_DIR / "kami" / "data" / ".pipeline" / "scheduler.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"Scheduled update launcher failed ({type(exc).__name__}).\n",
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
