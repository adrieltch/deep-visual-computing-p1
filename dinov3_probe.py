"""Frozen Meta DINO features and a training-source logistic probe.

The train command never reads HAM10000. The predict command loads a classifier
fixed by train and reads test images only to produce the submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, functional as vision_functional
from transformers import AutoImageProcessor, AutoModel

from skin_baseline.data import SOURCES, make_split
from train import device_from_arg
from validate_submission import validate_submission


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
MODEL_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
FEATURE_RECIPE = "last_hidden_state:cls_plus_mean_patch;exclude_register_tokens"
BACKBONES = {
    "dinov3-small": (MODEL_ID, MODEL_REVISION),
    "dinov2-small": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
}


class ImageRows(Dataset):
    def __init__(self, frame: pd.DataFrame, processor):
        self.paths = frame.image_path.tolist()
        self.ids = frame.ID.tolist()
        self.processor = processor

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
            size = getattr(self.processor, "size", None) or {}
            if (
                getattr(self.processor, "do_resize", False)
                and "shortest_edge" in size
                and self.processor.resample == Image.Resampling.BICUBIC
            ):
                # Match DINOv2's own bicubic resize before converting large images to arrays.
                rgb = vision_functional.resize(
                    rgb, size["shortest_edge"], interpolation=InterpolationMode.BICUBIC
                )
                pixels = self.processor(images=rgb, do_resize=False, return_tensors="pt")["pixel_values"][0]
            else:
                pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        return self.ids[index], pixels


def load_backbone(data_root: Path, device: torch.device, model_id: str, revision: str):
    cache = data_root / ".cache" / "huggingface"
    try:
        processor_options = {"use_fast": False} if model_id == BACKBONES["dinov2-small"][0] else {}
        processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision, cache_dir=cache, **processor_options
        )
        model = AutoModel.from_pretrained(
            model_id, revision=revision, cache_dir=cache, use_safetensors=True
        )
    except (GatedRepoError, OSError) as error:
        if "gated" not in str(error).lower() and not isinstance(error, GatedRepoError):
            raise
        raise RuntimeError(
            f"Meta's official DINO weights require approved access to {model_id}. "
            "Accept the terms on its Hugging Face page and run "
            "`hf auth login` locally before retrying."
        ) from error
    model.requires_grad_(False).eval().to(device)
    return processor, model


@torch.inference_mode()
def extract_features(
    frame: pd.DataFrame,
    processor,
    model,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        ImageRows(frame, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    arrays = []
    seen = 0
    start = time.monotonic()
    for ids, pixels in loader:
        outputs = model(pixel_values=pixels.to(device))
        tokens = outputs.last_hidden_state
        patch_start = 1 + getattr(model.config, "num_register_tokens", 0)
        if tokens.shape[1] <= patch_start:
            raise ValueError("DINO returned no image patch tokens")
        features = torch.cat((tokens[:, 0], tokens[:, patch_start:].mean(dim=1)), dim=1)
        arrays.append(features.float().cpu().numpy())
        seen += len(ids)
        if seen % (batch_size * 100) < len(ids) or seen == len(frame):
            print(
                json.dumps({"event": "features", "done": seen, "total": len(frame),
                            "elapsed_minutes": round((time.monotonic() - start) / 60, 2)}),
                flush=True,
            )
    return np.concatenate(arrays, axis=0)


def make_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def scores(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(targets, probabilities)),
        "accuracy": float(accuracy_score(targets, probabilities >= 0.5)),
    }


def fit(args: argparse.Namespace) -> None:
    root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_from_arg(args.device)

    frame = make_split(root, seed=args.seed, fold=args.fold)
    frame[["ID", "target", "source", "group_id", "split"]].to_csv(
        output / "split.csv", index=False
    )
    model_id, revision = BACKBONES[args.backbone]
    processor, model = load_backbone(root, device, model_id, revision)
    features = extract_features(frame, processor, model, device, args.batch_size, args.workers)
    labels = frame.target.to_numpy(dtype=np.int64)
    train_mask = frame.split.eq("train").to_numpy()
    val_mask = ~train_mask
    probe = make_classifier().fit(features[train_mask], labels[train_mask])
    val_probabilities = probe.predict_proba(features[val_mask])[:, 1]
    validation = scores(labels[val_mask], val_probabilities)
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    by_source = {}
    for source in sorted(val_frame.source.unique()):
        mask = val_frame.source.eq(source).to_numpy()
        by_source[source] = {
            "count": int(mask.sum()),
            **scores(labels[val_mask][mask], val_probabilities[mask]),
        }

    source_holdouts = {}
    if args.source_holdouts:
        for source in SOURCES:
            holdout_mask = frame.source.eq(source).to_numpy()
            heldout_probe = make_classifier().fit(features[~holdout_mask], labels[~holdout_mask])
            heldout_probabilities = heldout_probe.predict_proba(features[holdout_mask])[:, 1]
            source_holdouts[source] = {
                "count": int(holdout_mask.sum()),
                **scores(labels[holdout_mask], heldout_probabilities),
            }

    # Refit the same, already-fixed pipeline on all permitted training rows.
    final_probe = make_classifier().fit(features, labels)
    config = {
        "model_id": model_id,
        "revision": revision,
        "feature_recipe": FEATURE_RECIPE,
        "preprocessing": processor.to_dict(),
        "classifier": "StandardScaler + LogisticRegression(C=1.0, max_iter=1000)",
        "selection": "fixed recipe; held-out validation for comparison only",
        "seed": args.seed,
        "fold": args.fold,
        "train_count": int(train_mask.sum()),
        "validation_count": int(val_mask.sum()),
        "final_fit_count": len(frame),
        "validation": validation,
        "by_source": by_source,
        "source_holdouts": source_holdouts,
        "mean_source_holdout_auc": float(np.mean([row["auc"] for row in source_holdouts.values()])) if source_holdouts else None,
        "torch": torch.__version__,
    }
    manifest_bytes = (output / "split.csv").read_bytes()
    config["split_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    joblib.dump({"classifier": final_probe, "config": config}, output / "probe.joblib")
    print(json.dumps({"event": "finished", "validation": validation, "by_source": by_source,
                      "source_holdouts": source_holdouts,
                      "classifier": str(output / "probe.joblib")}), flush=True)


def predict(args: argparse.Namespace) -> None:
    root = args.data_root.resolve()
    artifact = joblib.load(args.artifact)
    config = artifact["config"]
    if (config["model_id"], config["revision"]) not in BACKBONES.values() or config["feature_recipe"] != FEATURE_RECIPE:
        raise ValueError("Artifact does not match a fixed DINO feature recipe")
    frame = pd.read_csv(root / "test.csv", dtype={"ID": str})
    if list(frame.columns) != ["ID"] or frame.ID.isna().any() or frame.ID.duplicated().any():
        raise ValueError("test.csv must contain one unique ID column")
    frame["image_path"] = [str(root / "test_images" / "HAM10000" / f"{i}.jpg") for i in frame.ID]
    missing = frame.loc[~frame.image_path.map(lambda p: Path(p).is_file()), "ID"]
    if len(missing):
        raise FileNotFoundError(f"Missing test images: {missing.head(5).tolist()}")

    torch.set_num_threads(args.cpu_threads)
    device = device_from_arg(args.device)
    processor, model = load_backbone(root, device, config["model_id"], config["revision"])
    if processor.to_dict() != config["preprocessing"]:
        raise ValueError("Image processor differs from the one used during fitting")
    features = extract_features(frame, processor, model, device, args.batch_size, args.workers)
    probabilities = artifact["classifier"].predict_proba(features)[:, 1]
    result = pd.DataFrame({"ID": frame.ID, "probability": probabilities})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.8f")
    count = validate_submission(root / "test.csv", args.output)
    print(f"Wrote {count} predictions to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="Fit using official training sources only")
    fit_parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov3_probe"))
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.add_argument("--fold", type=int, default=0)
    fit_parser.add_argument("--backbone", choices=tuple(BACKBONES), default="dinov3-small")
    fit_parser.add_argument("--source-holdouts", action="store_true", help="Also score each whole training source held out")
    predict_parser = subparsers.add_parser("predict", help="Run the fixed model on test images")
    predict_parser.add_argument("--artifact", type=Path, default=Path("outputs/dinov3_probe/probe.joblib"))
    predict_parser.add_argument("--output", type=Path, default=Path("outputs/dinov3_probe/submission.csv"))
    for subparser in (fit_parser, predict_parser):
        subparser.add_argument("--data-root", type=Path, default=Path("."))
        subparser.add_argument("--batch-size", type=int, default=8)
        subparser.add_argument("--workers", type=int, default=2)
        subparser.add_argument("--cpu-threads", type=int, default=4)
        subparser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive; workers cannot be negative")
    if args.command == "fit":
        fit(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
