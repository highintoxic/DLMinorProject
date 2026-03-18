"""
models.py — Transfer learning model builders (Task 5)
"""

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
    output = layers.Dense(num_classes, activation="softmax")(x)

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
