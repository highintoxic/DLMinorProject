"""
preprocessing.py — Image preprocessing and dataset splitting (Task 4)
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


def preprocess_dataset(
    class_images: dict[str, list],
    target_size: tuple[int, int] = (224, 224),
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Task 4: Resize to 224×224, normalize to [0, 1], and split into
    Train (70%) / Validation (15%) / Test (15%).

    Args:
        class_images: dict of class_name -> list of Paths or numpy arrays.

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        class_names, label_map
    """
    paths: list[str] = []
    labels: list[int] = []
    class_names = sorted(class_images.keys())
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for cls, items in class_images.items():
        lbl = label_map[cls]
        for item in items:
            paths.append(str(item))
            labels.append(lbl)

    X = np.array(paths)
    y = np.array(labels)

    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Second split: separate train and validation from remaining
    relative_val = val_size / (1 - test_size)  # ~0.176
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=seed, stratify=y_temp
    )

    print(f"Dataset splits:")
    print(f"  Train:      {X_train.shape[0]:>5} samples")
    print(f"  Validation: {X_val.shape[0]:>5} samples")
    print(f"  Test:       {X_test.shape[0]:>5} samples")
    print(f"  Classes:    {len(class_names)}")
    print(f"  Image size: {target_size[0]}×{target_size[1]}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "label_map": label_map,
    }
