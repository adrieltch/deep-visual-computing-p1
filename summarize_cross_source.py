"""Summarize source-held-out validation from official training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skin_baseline.data import SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/cross_source"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--at-epoch", type=int, help="Compare a fixed epoch across folds, including incomplete runs")
    args = parser.parse_args()
    if args.at_epoch is not None and args.at_epoch < 1:
        parser.error("--at-epoch must be positive")

    reports = []
    for sampling in ("natural", "balanced"):
        folds = []
        for source in SOURCES:
            path = args.root / f"{source}-{sampling}" / "history.json"
            if not path.is_file():
                print(f"Pending: {path}")
                continue
            history = json.loads(path.read_text())
            required = args.at_epoch if args.at_epoch is not None else args.epochs
            if len(history) < required or (args.at_epoch is None and len(history) != args.epochs):
                print(f"Incomplete ({len(history)}/{args.epochs} epochs): {path}")
                continue
            best = history[args.at_epoch - 1] if args.at_epoch is not None else max(
                history, key=lambda row: row["validation"]["auc"]
            )
            folds.append({
                "held_out_source": source,
                "count": best["by_source"][source]["count"],
                "epoch": best["epoch"],
                "auc": best["validation"]["auc"],
                "accuracy": best["validation"]["accuracy"],
            })
        report = {"sampling": sampling, "selection": f"epoch {args.at_epoch}" if args.at_epoch is not None else "best held-out AUC", "folds": folds}
        if len(folds) == len(SOURCES):
            report["mean_auc"] = float(np.mean([fold["auc"] for fold in folds]))
            report["mean_accuracy"] = float(np.mean([fold["accuracy"] for fold in folds]))
            if args.at_epoch is None:
                report["median_best_epoch"] = int(np.median([fold["epoch"] for fold in folds]))
        reports.append(report)
        print(json.dumps(report, indent=2))
    if all(len(report["folds"]) == len(SOURCES) for report in reports):
        filename = f"epoch{args.at_epoch}_summary.json" if args.at_epoch is not None else "summary.json"
        (args.root / filename).write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
