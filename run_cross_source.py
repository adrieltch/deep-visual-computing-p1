"""Run and resume the six MobileNet source-held-out experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from skin_baseline.data import SOURCES


def is_complete(directory: Path, source: str, sampling: str, epochs: int) -> bool:
    config_path = directory / "config.json"
    history_path = directory / "history.json"
    checkpoint_path = directory / "best.pt"
    if not all(path.is_file() for path in (config_path, history_path, checkpoint_path)):
        return False
    config = json.loads(config_path.read_text())
    history = json.loads(history_path.read_text())
    return (
        config.get("holdout_source") == source
        and config.get("source_sampling") == sampling
        and config.get("epochs") == epochs
        and len(history) == epochs
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cross_source"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        parser.error("epochs must be positive and workers cannot be negative")

    for source in SOURCES:
        for sampling in ("natural", "balanced"):
            directory = (args.output_root / f"{source}-{sampling}").resolve()
            if is_complete(directory, source, sampling, args.epochs):
                print(f"Already complete: {directory}", flush=True)
                continue
            print(f"Running: held out {source}, {sampling} sampling", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("train.py")),
                    "--data-root", str(args.data_root.resolve()),
                    "--output-dir", str(directory),
                    "--holdout-source", source,
                    "--source-sampling", sampling,
                    "--epochs", str(args.epochs),
                    "--workers", str(args.workers),
                    "--device", args.device,
                ],
                check=True,
            )
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("summarize_cross_source.py")),
         "--root", str(args.output_root.resolve()), "--epochs", str(args.epochs)],
        check=True,
    )


if __name__ == "__main__":
    main()
