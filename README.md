# SKIN Generalization

An image-only binary malignancy classifier for the SKIN Generalization educational competition. It trains on the 22,894 labeled BCN20000, MSK-1, and MSK-2 rows in `train.csv`. HAM10000 is used only by `predict.py` after a checkpoint has been selected on training-source validation data. This model is not for clinical use.

## Setup on this Mac

The local Conda environment lives at `.conda/skin-baseline` and is excluded from Git. To recreate it:

```sh
conda env create --prefix ./.conda/skin-baseline --file environment.yml
conda activate "$PWD/.conda/skin-baseline"
```

The data directory should contain `train.csv`, `test.csv`, `metadata/original/`, `train_images/{BCN20000,MSK-1,MSK-2}/`, and `test_images/HAM10000/`. Raw data and generated artifacts are excluded from Git.

## Run

From the project root:

```sh
python train.py --data-root . --output-dir outputs/baseline --epochs 5 --workers 2 --device auto
python predict.py --data-root . --checkpoint outputs/baseline/best.pt --output outputs/baseline/submission.csv --workers 2 --device auto
python validate_submission.py outputs/baseline/submission.csv
```

Training writes `split.csv`, `config.json`, `history.json`, and the checkpoint with the highest validation ROC AUC to `outputs/baseline/`. The split is fixed by seed 42 and fold 0 of a five-fold stratified group split. The strata combine source and binary target. Rows sharing a patient ID stay together; if no patient ID exists, rows sharing a lesion ID stay together. Rows without either identifier are grouped by image ID. In the supplied labeled MSK-2 data, 1,618 rows lack patient IDs and cannot be grouped beyond their image ID using the supplied metadata.

The checkpoint stores the epoch, validation AUC and accuracy at probability 0.5, training configuration, and model state. The model uses a 224-pixel training crop, horizontal and vertical flips, and mild color jitter. Validation and inference resize to 256 pixels and center crop to 224, then normalize with the ImageNet mean and standard deviation. The training loss is binary cross entropy with logits; probabilities come from a sigmoid at inference.

`predict.py` reads only the fixed checkpoint, `test.csv`, and test images. The submission checker enforces the exact columns, complete unique ID set, and finite probabilities in `[0, 1]`.

### First run (2026-10-03)

The five-epoch run on this M2 Mac selected epoch 2 from 18,372 training and 4,522 validation images. Held-out training-source validation ROC AUC was **0.8599** and accuracy at threshold 0.5 was **0.7682**. ROC AUC by source was 0.8637 (BCN20000), 0.8311 (MSK-1), and 0.8264 (MSK-2). These are validation results, not HAM10000 scores. The local `outputs/baseline/best.pt` SHA-256 is `9588bdb7670d36157f3932a9112de029cb18ae90369fd420805c59157268b88d`. Fixed inference produced `outputs/baseline/submission.csv`; the independent checker accepted all 10,015 unique test IDs and finite probabilities. Its SHA-256 is `c5eba123b35704e4576b3f235b69b6f6d2ddbbbcfceb5bf50f15406c3b9f4167`.

## Pretraining disclosure

- Architecture: TorchVision `mobilenet_v3_small`, version 0.22.0.
- Initial weights: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, trained on ImageNet-1K with TorchVision's published recipe. The weight URL and SHA-256 are recorded in `outputs/baseline/config.json`.
- TorchVision source code license: BSD 3-Clause. ImageNet data terms restrict use to non-commercial research and education. TorchVision cautions that pretrained weights can inherit data terms; this competition is educational and research-only.
- No skin-lesion or HAM10000 pretrained weights are used.

Primary sources: [TorchVision model and preprocessing](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html), [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE), and [ImageNet terms of access](https://www.image-net.org/download.php).

## Scope of the initial baseline

This is one image-only baseline and one training-source validation split. The model is selected solely by its training-source validation ROC AUC. The validation score estimates performance on held-out groups from the supplied sources; it does not establish external HAM10000 performance. The extra images in the source folders that do not appear in `train.csv` are unused.

## Source-held-out validation

The initial group split contains all three sources in both training and
validation. To estimate performance when the institution changes, run each
of BCN20000, MSK-1, and MSK-2 as a wholly held-out validation source. The
runner compares the original source mix (`natural`) with equal expected
sampling of the two training sources (`balanced`). The balanced sampler uses
inverse training-source frequency with replacement. Both variants use the
same architecture, augmentations, optimizer, seed, and three-epoch budget.

```sh
python run_cross_source.py --data-root . --output-root outputs/cross_source --epochs 3 --workers 2 --device auto
python summarize_cross_source.py --root outputs/cross_source --epochs 3
```

The runner skips completed folds and can be rerun after an interruption. Each
fold writes its own split manifest, configuration, history, and AUC-selected
checkpoint. The summary reports each held-out source's best AUC and accuracy
at the fixed 0.5 threshold, plus the unweighted mean over the three sources.
Only BCN/MSK images and labels are used in this experiment. These validation
results can guide development; HAM10000 leaderboard scores cannot.

### Training-source comparison (2026-10-03)

| Recipe | BCN20000 AUC | MSK-1 AUC | MSK-2 AUC | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| MobileNet, natural sampling, epoch 1 | 0.7740 | 0.7492 | 0.8159 | **0.7797** |
| MobileNet, balanced sampling, epoch 1 | 0.7655 | 0.7464 | 0.7841 | 0.7653 |
| Frozen DINOv2-S + logistic classifier | 0.7269 | 0.7352 | 0.7960 | 0.7527 |
| MobileNet, natural sampling, best epoch per held-out source | 0.7740 | 0.8049 | 0.8159 | 0.7982 |

The first-epoch MobileNet rows are a matched comparison. Natural sampling won
on all three held-out sources. The balanced MSK-2 run was stopped after its
first epoch because of high image-loading cost; its recorded first-epoch AUC
completes the matched comparison. The best-per-source row selects a checkpoint
using each held-out source, so it is a more optimistic estimate. Its best
epochs were 1, 2, and 1; their median is 1. That is the fixed epoch count for
the final natural-sampling MobileNet fit on all 22,894 official training rows.
The DINOv2 classifier used only the other two official sources for each
held-out source and was not selected. The comparison is recorded in
`outputs/cross_source/epoch1_summary.json` and
`outputs/dinov2_probe/config.json`.

These numbers are training-source validation results. They do not establish
whether the next submission will improve on HAM10000.

### Selected all-training candidate

The selected recipe is MobileNetV3-Small, natural source sampling, and one
training epoch. It was fitted on all 22,894 official labeled rows. The final
fit has no validation split and uses the epoch count fixed above.

```sh
python train.py --data-root . --output-dir outputs/mobilenet_source_cv_final --full-data --source-sampling natural --epochs 1 --workers 2 --device auto
python predict.py --data-root . --checkpoint outputs/mobilenet_source_cv_final/final.pt --output outputs/mobilenet_source_cv_final/submission.csv --workers 2 --device auto
```

The checkpoint SHA-256 is
`1c82161f80a18c8046711dfa0f541e2c3b1487a289aeb03b6f2bf078e9972f85`.
The submission SHA-256 is
`d68540db09e5d9d43fa9dfe31f841347f835829a3d7f168cb11c3c3a7b2463d5`.
The submission checker accepted exactly 10,015 unique IDs and finite
probabilities in `[0, 1]`. The test images were used only after the recipe was
fixed from official training-source validation.

## Frozen DINOv3 ViT-S/16 probe (pending weight access)

`dinov3_probe.py` is a second, separate experiment. It uses Meta's official
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
at revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`. The backbone is frozen;
its CLS token and mean image-patch token are concatenated, standardized, and fed
to a logistic regression classifier (`C=1.0`). The four register tokens are
excluded from the patch mean. Image preprocessing is loaded from the model's
own processor. The same seed-42/fold-0 grouped BCN/MSK split gives a direct
training-source comparison with the MobileNet baseline; validation AUC is the
primary comparison, with accuracy at the fixed 0.5 threshold reported too.
The fixed classifier recipe is then refit on all permitted training rows.

Meta gates these weights behind an access agreement. You must request access
on the model page and authenticate locally with `hf auth login` (or a local
`HF_TOKEN` environment variable); do not put a token in Git or share it in
chat. Install the updated environment dependencies, then run:

```sh
python dinov3_probe.py fit --data-root . --output-dir outputs/dinov3_probe --device auto
python dinov3_probe.py predict --data-root . --artifact outputs/dinov3_probe/probe.joblib --output outputs/dinov3_probe/submission.csv --device auto
```

Run `predict` only after the training-source comparison and the inference
procedure are fixed. It does not read HAM10000 labels or use HAM10000 to tune
the model. `fit` writes the split manifest, configuration, validation metrics,
and trained classifier to the ignored `outputs/dinov3_probe/` directory.
No DINOv3 validation score or submission has been produced yet because this Mac
does not currently have access to the official weights.

For final disclosure: Meta DINOv3 ViT-S/16 is a 21.6M-parameter general-purpose
visual foundation model. Meta reports pretraining on its curated LVD-1689M
collection of public Instagram images; the complete corpus is not public.
Its weights are distributed under the
[DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md),
which must be reviewed before use. No HAM10000-specific pretrained model is
part of this experiment. See the [official model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
and [Transformers DINOv3 documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dinov3).

### Public DINOv2 alternative

The same frozen-feature probe also supports Meta's public
[`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) at
revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. It uses the model's
own image processor and concatenates the CLS and mean patch embeddings. A
logistic classifier is fitted separately on the other two official training
sources for each source-held-out AUC, then the fixed recipe is refitted on all
official training rows for inference. No HAM10000 information is used in fitting
or choosing the probe.

```sh
python dino_probe.py fit --backbone dinov2-small --source-holdouts --data-root . --output-dir outputs/dinov2_probe --device auto
```

If it is selected on training-source validation, its fixed classifier can be
used for inference:

```sh
python dino_probe.py predict --artifact outputs/dinov2_probe/probe.joblib --output outputs/dinov2_probe/submission.csv --device auto
```

The DINOv2 probe was not selected in the 2026-10-03 training-source comparison.

Meta reports that DINOv2 ViT-S/14 was pretrained without labels on LVD-142M;
the complete image corpus is not public. Its code and model weights are
distributed under Apache 2.0. See the
[official model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
and [license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
