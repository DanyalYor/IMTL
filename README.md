## IMTL – Food-101 Classification with EfficientNet-B0

This project trains and evaluates an EfficientNet-B0 classifier on a Food-101-style dataset using PyTorch.
It supports multiple training modes (scratch / feature extraction / full fine-tuning / partial fine-tuning), multiple augmentation presets, W&B logging, and optional Ray Tune hyperparameter search.

Project Features:

- EfficientNet-B0 backbone (torchvision.models.efficientnet_b0)

Training modes:

- From scratch

- Baseline (pretrained, no training)

- Feature extractor (train classifier head only)

- Full fine-tuning (train all layers)

- Partial fine-tuning (freeze early blocks)

Augmentation presets:

- Minimal

- Geometric

- White-filter (solarize)

- Photometric (ColorJitter)

- AutoAugment (ImageNet policy)

- Combo (strong aug + erasing)
Metrics:

- Train/Val/Test loss + accuracy

- Confusion matrix (101×101)

- Top confusions table

- Per-class accuracy table (sorted)

- Misclassified image grids

- Grad-CAM for wrong predictions

Optional: Ray Tune hyperparameter search

### Requirements

Install dependencies:

```
pip install -r requirements.txt
```

### Dataset

The dataset is publicly available from the original authors:
Food-101 dataset (official):

👉 <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>
