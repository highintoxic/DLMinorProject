"""
sampling.py — Under-sampling and over-sampling (augmentation) strategies (Tasks 2 & 3)
"""

import random
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def undersample(
    class_images: dict[str, list[Path]],
    threshold: int = 200,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """
    Task 2: Randomly remove images from majority classes so each has at most
    `threshold` images. Minority classes (≤ threshold) are untouched.

    Returns:
        A new dict with under-sampled image paths.
    """
    rng = random.Random(seed)
    result: dict[str, list[Path]] = {}
    for cls, paths in class_images.items():
        if len(paths) > threshold:
            result[cls] = rng.sample(paths, threshold)
        else:
            result[cls] = list(paths)  # keep all
    return result


def _augment_image(img_array: np.ndarray, datagen: ImageDataGenerator) -> np.ndarray:
    """Apply a random augmentation to a single image array (H, W, C)."""
    img_4d = img_array.reshape((1,) + img_array.shape)
    aug_iter = datagen.flow(img_4d, batch_size=1)
    return next(aug_iter)[0].astype(np.uint8)


def oversample_augment(
    class_images: dict[str, list[Path]],
    threshold: int = 200,
    target_size: tuple[int, int] = (224, 224),
    save_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, list]:
    """
    Task 3: Apply augmentations to minority classes to bring them up to `threshold`.

    Augmentations:
        - Horizontal flip
        - Rotation ±20°
        - Width/height shift ±10%
        - Zoom ±10%

    If `save_dir` is provided, augmented images are saved to disk and the returned
    dict contains file paths. Otherwise it contains numpy arrays.

    Returns:
        dict mapping class_name -> list of paths (or arrays) with length >= threshold.
    """
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent.parent / "data" / "augmented"

    rng = random.Random(seed)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    result: dict[str, list] = {}
    for cls, paths in class_images.items():
        current = list(paths)
        needed = threshold - len(current)

        if needed <= 0:
            result[cls] = current
            continue

        # Generate augmented images
        augmented: list = []
        for i in range(needed):
            src_path = rng.choice(paths)
            img = Image.open(src_path).convert("RGB").resize(target_size)
            img_arr = np.array(img)
            aug_arr = _augment_image(img_arr, datagen)

            if save_dir:
                cls_dir = save_dir / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                aug_path = cls_dir / f"aug_{i:04d}.jpg"
                Image.fromarray(aug_arr).save(aug_path)
                augmented.append(aug_path)
            else:
                augmented.append(aug_arr)

        result[cls] = current + augmented

    return result


def show_augmentation_samples(
    image_path: Path,
    n_samples: int = 5,
    target_size: tuple[int, int] = (224, 224),
    save_path: str | None = None,
) -> None:
    """Display original image alongside n augmented versions."""
    import matplotlib.pyplot as plt

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    img = Image.open(image_path).convert("RGB").resize(target_size)
    img_arr = np.array(img)

    fig, axes = plt.subplots(1, n_samples + 1, figsize=(3 * (n_samples + 1), 3))
    axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(n_samples):
        aug = _augment_image(img_arr, datagen)
        axes[i + 1].imshow(aug)
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=10)
        axes[i + 1].axis("off")

    plt.suptitle(f"Augmentation Samples — {image_path.parent.name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
