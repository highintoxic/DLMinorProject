<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Deep Learning WCE Classification Project — Overview

This is a **medical image classification lab project** where you build and compare deep learning models to detect gastrointestinal (GI) diseases from Wireless Capsule Endoscopy (WCE) images, with a strong focus on handling **class imbalance** — a critical challenge in medical datasets.

***

## The Core Problem

WCE datasets like Kvasir-Capsule are heavily imbalanced — some GI conditions (e.g., normal mucosa) have thousands of images, while rare diseases (e.g., bleeding, ulcers) have very few. If you train a model on raw data, it'll just predict the majority class and still get high accuracy — which is **dangerous in clinical settings**.

***

## What You Do — Task by Task

### Task 1 — Understand the Imbalance

- Load the **Kvasir-Capsule dataset** and plot a bar chart of how many images exist per class
- Identify which classes are over-represented (majority) and which are rare (minority)
- Write a short explanation of why imbalance is dangerous in medical diagnosis (e.g., missing a cancer case is far worse than a false alarm)


### Task 2 — Fix Imbalance via Under-Sampling

- Randomly **remove images from majority classes** until each has at most ~200 images
- Keep all minority class samples untouched
- Plot the updated distribution and note how much data you lost


### Task 3 — Fix Imbalance via Over-Sampling (Augmentation)

- Apply image augmentations **only to minority classes** to synthetically generate new samples:
- Horizontal flip, rotation ±20°, zoom, width/height shift
- Bring minority classes up to the same ~200 image threshold
- Show before/after sample images and updated dataset size


### Task 4 — Pre-Processing

- Resize all images to **224×224** (standard for most pretrained models)
- Normalize pixel values to ****
- Split into **Train (70%) / Validation (15%) / Test (15%)**


### Task 5 — Transfer Learning (3 Models)

Pick **three** architectures (e.g., EfficientNet-B0, MobileNet, ResNet101v2) and for each:

- Load ImageNet pretrained weights
- **Freeze early layers** (keep learned low-level features)
- Replace the final layer with your own classification head (number of GI classes)
- Add **Dropout + L2 regularization** to reduce overfitting
- Compare model summaries: trainable vs frozen parameters


### Task 6 — Smart Learning Rate Control

Instead of a fixed learning rate, implement adaptive strategies:

- **ReduceLROnPlateau** — cuts LR when validation loss stops improving
- **Cosine Decay** — smoothly decreases LR over epochs
- Plot LR vs epoch and training/validation loss curves


### Task 7 — Train \& Compare Under 3 Settings

Train your models three times each:


| Setting | Description |
| :-- | :-- |
| Baseline | Raw imbalanced data, no handling |
| Under-sampling | Majority classes trimmed to threshold |
| Under-sampling + Augmentation | Balanced via both techniques |

Evaluate each with **Accuracy, Precision, Recall, F1-score**, and a **Confusion Matrix**. Write 8–10 lines analyzing which setting performed best and why.

***

## Key Deliverables Checklist

- Bar charts of class distribution (before \& after balancing)
- Augmentation sample image grid
- Model summaries for 3 architectures
- LR vs epoch + loss curves plots
- Comparison table of metrics across 3 training settings
- Confusion matrices
- Short written analyses for Tasks 1 \& 7

***

## Recommended Tech Stack

- **Framework:** TensorFlow/Keras or PyTorch
- **Models:** `tf.keras.applications.EfficientNetB0`, `MobileNetV2`, `ResNet101V2`
- **Augmentation:** `ImageDataGenerator` (Keras) or `torchvision.transforms`
- **Metrics:** `sklearn.metrics` — classification_report, confusion_matrix
- **Dataset:** Download Kvasir-Capsule from [osf.io/dv2ag](https://osf.io/dv2ag/)

The biggest conceptual takeaway is understanding **why standard accuracy fails for imbalanced medical data** and how combining under-sampling + augmentation + transfer learning produces a clinically meaningful model.

