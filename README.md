# SKIN Generalization: first baseline

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

## Scope of this iteration

This is one image-only baseline and one training-source validation split. The model is selected solely by its training-source validation ROC AUC. The validation score estimates performance on held-out groups from the supplied sources; it does not establish external HAM10000 performance. The extra images in the source folders that do not appear in `train.csv` are unused.

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
