"""
models.py — Transfer learning model builders (Task 5)
"""

import gc

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet101V2


# ── Registry of supported architectures ──────────────────────────────
ARCHITECTURES = {
    "EfficientNetB0": EfficientNetB0,
    "MobileNetV2": MobileNetV2,
    "ResNet101V2": ResNet101V2,
}


def build_model(
    arch: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    freeze_ratio: float = 0.8,
    dropout_rate: float = 0.3,
    l2_weight: float = 1e-4,
) -> tf.keras.Model:
    """
    Task 5: Build a transfer-learning model.

    - Loads ImageNet pretrained weights
    - Freezes `freeze_ratio` fraction of early layers
    - Adds a classification head with Dropout + L2 regularization

    Args:
        arch: One of 'EfficientNetB0', 'MobileNetV2', 'ResNet101V2'
        num_classes: Number of output classes
        freeze_ratio: Fraction of base layers to freeze (0.0 = all trainable)
        dropout_rate: Dropout probability before final dense layer
        l2_weight: L2 regularization strength

    Returns:
        Compiled tf.keras.Model
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(ARCHITECTURES.keys())}")

    # Enable Mixed Precision for faster training on modern GPUs (like Colab T4)
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    except Exception as e:
        print("Mixed precision could not be enabled:", e)

    base_model = ARCHITECTURES[arch](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze early layers
    num_freeze = int(len(base_model.layers) * freeze_ratio)
    for layer in base_model.layers[:num_freeze]:
        layer.trainable = False
    for layer in base_model.layers[num_freeze:]:
        layer.trainable = True

    # Classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_weight),
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    # Ensure final layer is float32 for numerical stability in mixed precision
    output = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs=base_model.input, outputs=output, name=arch)
    return model


def print_model_summary(model: tf.keras.Model) -> None:
    """Print a compact summary showing trainable vs frozen parameters."""
    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    frozen = total - trainable

    print(f"\n{'='*60}")
    print(f"  Model: {model.name}")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {frozen:>12,}")
    print(f"  Trainable ratio:      {trainable/total:>11.1%}")
    print(f"{'='*60}\n")


def build_all_models(num_classes: int, **kwargs) -> dict[str, tf.keras.Model]:
    """Build all three architectures and return as a dict."""
    model_dict = {}
    for arch in ARCHITECTURES:
        print(f"Building {arch}...")
        model = build_model(arch, num_classes, **kwargs)
        print_model_summary(model)
        model_dict[arch] = model
    return model_dict


def clear_model_session() -> None:
    """Release GPU/CPU memory after a model training run.

    Call this between training different architectures to free weights,
    optimizer state, and TF graph nodes from the previous model.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    print("TF session cleared and garbage collected.")
