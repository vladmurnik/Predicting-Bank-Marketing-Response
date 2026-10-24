"""Cross-platform task runner: works even where GNU make is not installed."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS: dict[str, list[list[str]]] = {
    "setup": [["uv", "sync"]],
    "download": [["uv", "run", "python", "-m", "bank_marketing.cli", "download"]],
    "repro": [["uv", "run", "python", "-m", "bank_marketing.cli", "all"]],
    "test": [["uv", "run", "pytest", "-q"]],
    "lint": [["uv", "run", "ruff", "check", "."]],
    "serve": [
        [
            "uv",
            "run",
            "uvicorn",
            "bank_marketing.serve:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    ],
}

CLEAN_TARGETS = [
    ROOT / "reports" / "figures",
    ROOT / "reports" / "metrics.json",
    ROOT / "reports" / "run.json",
    ROOT / "reports" / "report.md",
    ROOT / "models" / "model.joblib",
]


def _run(command: list[str]) -> int:
    print("$ " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


def clean() -> int:
    for path in CLEAN_TARGETS:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"removed {path}")
        elif path.exists():
            path.unlink()
            print(f"removed {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Project tasks without GNU make")
    parser.add_argument("task", choices=[*TASKS, "clean"])
    args = parser.parse_args()

    if args.task == "clean":
        return clean()

    for command in TASKS[args.task]:
        code = _run(command)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
