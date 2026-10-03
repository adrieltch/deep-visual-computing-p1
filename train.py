"""Train the SKIN Generalization MobileNet baseline and source-holdout variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import __version__ as torchvision_version
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from skin_baseline.data import SOURCES, LesionDataset, make_split


WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(pretrained: bool) -> nn.Module:
    model = mobilenet_v3_small(weights=WEIGHTS if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def metrics(labels: list[float], probabilities: list[float]) -> dict[str, float]:
    values = {"accuracy": float(accuracy_score(labels, np.asarray(probabilities) >= 0.5))}
    values["auc"] = float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else math.nan
    return values


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, sources: list[str]):
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    for images, targets in loader:
        logits = model(images.to(device)).squeeze(1)
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(targets.tolist())
    overall = metrics(labels, probabilities)
    by_source = {}
    for source in sorted(set(sources)):
        indices = [i for i, value in enumerate(sources) if value == source]
        by_source[source] = {"count": len(indices), **metrics([labels[i] for i in indices], [probabilities[i] for i in indices])}
    return overall, by_source


def train(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(data_root / ".cache" / "torch"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = device_from_arg(args.device)

    frame = make_split(data_root, seed=args.seed, fold=args.fold)
    if args.full_data:
        frame["split"] = "train"
    elif args.holdout_source is not None:
        frame["split"] = np.where(frame.source.eq(args.holdout_source), "val", "train")
    manifest = frame[["ID", "target", "source", "group_id", "split"]]
    manifest.to_csv(output_dir / "split.csv", index=False)
    training = frame.loc[frame.split == "train"].reset_index(drop=True)
    validation = frame.loc[frame.split == "val"].reset_index(drop=True)
    if training.empty or (validation.empty and not args.full_data):
        raise ValueError("The group split produced an empty partition")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.source_sampling == "balanced":
        source_counts = training.source.value_counts().to_dict()
        weights = training.source.map(lambda source: 1.0 / source_counts[source]).to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(training), replacement=True, generator=generator)
    train_loader = DataLoader(
        LesionDataset(training, training=True),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    val_loader = None if args.full_data else DataLoader(
        LesionDataset(validation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    model = make_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    config = {
        "architecture": "torchvision.models.mobilenet_v3_small",
        "pretrained_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "pretrained_url": WEIGHTS.url,
        "pretraining_dataset": "ImageNet-1K",
        "input_size": 224,
        "seed": args.seed,
        "fold": args.fold,
        "holdout_source": args.holdout_source,
        "source_sampling": args.source_sampling,
        "full_data": args.full_data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "train_count": len(training),
        "validation_count": len(validation),
        "torch": str(torch.__version__),
        "torchvision": torchvision_version,
    }
    weight_file = data_root / ".cache" / "torch" / "checkpoints" / WEIGHTS.url.rsplit("/", 1)[-1]
    if weight_file.is_file():
        config["pretrained_sha256"] = hashlib.sha256(weight_file.read_bytes()).hexdigest()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"event": "start", **config}), flush=True)

    best_auc = -math.inf
    history = []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device=device, dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()
        if args.full_data:
            result = {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "elapsed_minutes": round((time.monotonic() - start) / 60, 2),
            }
            history.append(result)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            if epoch == args.epochs:
                checkpoint = {"model_state": model.state_dict(), "epoch": epoch, "config": config}
                temporary = output_dir / "final.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output_dir / "final.pt")
            continue
        overall, by_source = evaluate(model, val_loader, device, validation.source.tolist())
        result = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation": overall,
            "by_source": by_source,
            "elapsed_minutes": round((time.monotonic() - start) / 60, 2),
        }
        history.append(result)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if math.isfinite(overall["auc"]) and overall["auc"] > best_auc:
            best_auc = overall["auc"]
            checkpoint = {"model_state": model.state_dict(), "epoch": epoch, "validation": overall, "config": config}
            temporary = output_dir / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output_dir / "best.pt")
    if args.full_data:
        print(json.dumps({"event": "finished", "checkpoint": str(output_dir / "final.pt")}), flush=True)
    else:
        print(json.dumps({"event": "finished", "best_auc": best_auc, "checkpoint": str(output_dir / "best.pt")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--holdout-source", choices=tuple(SOURCES))
    parser.add_argument("--source-sampling", choices=("natural", "balanced"), default="natural")
    parser.add_argument("--full-data", action="store_true", help="Fit all official training rows for a fixed epoch count")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("epochs, batch size, and CPU threads must be positive; workers cannot be negative")
    if args.full_data and args.holdout_source is not None:
        parser.error("--full-data and --holdout-source cannot be combined")
    if (args.full_data or args.holdout_source is not None or args.source_sampling != "natural") and args.output_dir == Path("outputs/baseline"):
        parser.error("use a separate --output-dir for source-holdout, balanced, or full-data training")
    train(args)


if __name__ == "__main__":
    main()
