"""Run a fixed trained baseline on the HAM10000 test images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from skin_baseline.data import LesionDataset
from train import device_from_arg, make_model
from validate_submission import validate_submission


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    # Checkpoints written by early runs stored torch.__version__ as TorchVersion.
    # Allow only that known class while keeping PyTorch's restricted loader on.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    device = device_from_arg(args.device)
    torch.set_num_threads(args.cpu_threads)
    model = model.to(device).eval()

    test_csv = data_root / "test.csv"
    frame = pd.read_csv(test_csv, dtype={"ID": str})
    if list(frame.columns) != ["ID"] or frame["ID"].duplicated().any():
        raise ValueError("test.csv must contain one unique ID column")
    frame["image_path"] = [str(data_root / "test_images" / "HAM10000" / f"{image_id}.jpg") for image_id in frame.ID]
    missing = frame.loc[~frame.image_path.map(lambda p: Path(p).is_file()), "ID"]
    if len(missing):
        raise FileNotFoundError(f"Missing test images: {missing.head(5).tolist()}")
    loader = DataLoader(
        LesionDataset(frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "probability"])
        for images, image_ids in loader:
            logits = model(images.to(device)).squeeze(1)
            values = torch.sigmoid(logits).cpu().tolist()
            writer.writerows((image_id, f"{probability:.8f}") for image_id, probability in zip(image_ids, values))
    count = validate_submission(test_csv, args.output)
    print(f"Wrote {count} predictions to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive; workers cannot be negative")
    predict(args)


if __name__ == "__main__":
    main()
