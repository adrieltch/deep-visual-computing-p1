"""Check a SKIN Generalization submission against the provided test IDs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def validate_submission(test_csv: Path, submission_csv: Path) -> int:
    with test_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID"]:
            raise ValueError("test.csv must contain exactly the ID column")
        expected = [row["ID"] for row in reader]
    if len(expected) != len(set(expected)):
        raise ValueError("test.csv contains duplicate IDs")

    with submission_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "probability"]:
            raise ValueError("Submission columns must be exactly ID,probability")
        rows = list(reader)
    ids = [row["ID"] for row in rows]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Submission must contain each test ID exactly once")
    for row in rows:
        try:
            probability = float(row["probability"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid probability for {row['ID']}") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Probability out of range for {row['ID']}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-csv", type=Path, default=Path("test.csv"))
    args = parser.parse_args()
    count = validate_submission(args.test_csv, args.submission)
    print(f"Valid submission: {count} unique test IDs")


if __name__ == "__main__":
    main()
