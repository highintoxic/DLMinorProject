<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# What You Have to Do — Two Dataset Version

This project uses **Kvasir-Capsule** (imbalanced, main experiments) and **KVASIR v2** (balanced, for comparison) to build and evaluate deep learning models for GI disease classification.

***

## The Core Goal

Train 3 deep learning models to classify GI diseases from endoscopy images, handle the class imbalance problem using two techniques, and prove that your balancing methods actually work — by comparing against a naturally balanced dataset (KVASIR v2).

***

## What You Do — Step by Step

### Step 1 — Explore Both Datasets

- Plot class distribution of **Kvasir-Capsule** → show it's heavily imbalanced (14 classes, 47,238 images unevenly spread)
- Plot class distribution of **KVASIR v2** → show it's balanced (8 classes, 1,000 images each)
- Write why imbalance is dangerous in medical diagnosis


### Step 2 — Balance Kvasir-Capsule via Under-Sampling

- Cut majority classes down to **200 images max**
- Keep all minority class images untouched
- Show updated distribution plot


### Step 3 — Balance Further via Augmentation

- Apply flips, rotations, zoom, shifts **only to minority classes**
- Bring them up to 200 images
- Show before/after sample images


### Step 4 — Pre-Process Both Datasets

- Resize to **224×224**, normalize to ****[^1]
- Split both datasets: **70% train / 15% val / 15% test**


### Step 5 — Build 3 Transfer Learning Models

Pick 3 (e.g., EfficientNetB0, MobileNetV2, ResNet101V2):

- Load ImageNet weights
- Freeze early layers
- Add your own classification head with Dropout + L2
- Compare trainable vs frozen parameters across all 3


### Step 6 — Train with Smart Learning Rate

- Use **ReduceLROnPlateau** or **Cosine Decay**
- Plot LR vs epoch and loss curves


### Step 7 — Train \& Compare Under 4 Scenarios

| Scenario | Dataset Used | Purpose |
| :-- | :-- | :-- |
| Baseline (no handling) | Raw Kvasir-Capsule | Show how bad imbalance is |
| Under-sampling only | Kvasir-Capsule trimmed | Partial fix |
| Under-sampling + Augmentation | Kvasir-Capsule balanced | Full fix |
| Naturally balanced | KVASIR v2 | Gold standard comparison |

Evaluate all with **Accuracy, Precision, Recall, F1-score + Confusion Matrix**

***

## Final Deliverables

- 4 bar charts (class distributions)
- Augmentation sample image grid
- 3 model summaries (trainable vs frozen params)
- Loss + LR curves
- Comparison table (4 scenarios × 4 metrics)
- 3 confusion matrices
- Short written analysis

<div align="center">⁂</div>

[^1]: https://www.nature.com/articles/s41597-021-00920-z

