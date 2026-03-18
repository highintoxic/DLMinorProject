"""
data_utils.py — Dataset loading and class distribution visualization (Tasks 1)
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# ── Default dataset paths ─────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "labeled-images"         # Kvasir-Capsule
KVASIR_V2_DIR = Path(__file__).resolve().parent.parent / "data" / "kvasir-dataset-v2" # KVASIR v2


def load_image_paths(data_dir: Path = DATA_DIR) -> dict[str, list[Path]]:
    """
    Load all image paths organized by class label.

    Returns:
        dict mapping class_name -> list of image file paths
    """
    class_images: dict[str, list[Path]] = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
            class_images[class_dir.name] = images
    return class_images


def get_class_distribution(class_images: dict[str, list]) -> dict[str, int]:
    """Return a dict of class_name -> count."""
    return {k: len(v) for k, v in class_images.items()}


def plot_class_distribution(
    distribution: dict[str, int],
    title: str = "Class Distribution",
    save_path: str | None = None,
    figsize: tuple = (14, 6),
) -> None:
    """
    Plot a horizontal bar chart of class distribution.
    """
    classes = list(distribution.keys())
    counts = list(distribution.values())

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(classes))
    bars = ax.barh(classes, counts, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    ax.set_xlabel("Number of Images")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def load_image(path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load a single image, resize, and return as numpy array."""
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    return np.array(img)
