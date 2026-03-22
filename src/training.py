"""
training.py — Training loop, LR schedulers, and callbacks (Tasks 6 & 7)
"""

import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers


# ── Output directory ──────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def get_lr_callbacks(strategy: str = "reduce_on_plateau") -> list:
    """
    Task 6: Return LR-related callbacks.

    Args:
        strategy: 'reduce_on_plateau' or 'cosine_decay'
    """
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    if strategy == "reduce_on_plateau":
        cb_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3,
                min_lr=1e-6, verbose=1,
            )
        )
    # Cosine decay is set via the optimizer schedule (see compile_model)

    return cb_list


def compile_model(
    model: tf.keras.Model,
    lr_strategy: str = "reduce_on_plateau",
    initial_lr: float = 1e-3,
    total_epochs: int = 30,
    steps_per_epoch: int | None = None,
) -> tf.keras.Model:
    """
    Compile model with appropriate optimizer and LR schedule.
    """
    if lr_strategy == "cosine_decay":
        total_steps = (steps_per_epoch or 100) * total_epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=1e-6,
        )
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr_strategy: str = "reduce_on_plateau",
    save_name: str | None = None,
) -> tf.keras.callbacks.History:
    """
    Task 7: Train a model and return the training history.
    """
    cb_list = get_lr_callbacks(lr_strategy)

    # Save best model checkpoint
    if save_name:
        ckpt_dir = OUTPUT_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cb_list.append(
            callbacks.ModelCheckpoint(
                str(ckpt_dir / f"{save_name}_best.keras"),
                monitor="val_loss", save_best_only=True, verbose=1,
            )
        )

    # Check if inputs are file paths
    if isinstance(X_train[0], (str, np.str_)):
        def parse_function(filename, label):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # Perfect exact shuffling since array just contains file paths
        train_ds = train_ds.shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
        train_ds = train_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        # Prefetch capped at 2 to limit peak memory usage.
        train_ds = train_ds.batch(batch_size).prefetch(2)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        # NOTE: .cache() removed — it consumed ~1.3 GB of RAM and caused Colab crashes.
        # Prefetch capped at 2 to limit peak memory usage.
        val_ds = val_ds.batch(batch_size).prefetch(2)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=cb_list,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=1,
        )
    return history


def plot_training_history(
    history: tf.keras.callbacks.History,
    title: str = "Training History",
    save_path: str | None = None,
) -> None:
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_lr_schedule(
    initial_lr: float = 1e-3,
    total_epochs: int = 30,
    steps_per_epoch: int = 100,
    save_path: str | None = None,
) -> None:
    """Task 6: Plot cosine decay LR vs epoch."""
    total_steps = steps_per_epoch * total_epochs
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )

    steps = np.arange(total_steps)
    lrs = [schedule(s).numpy() for s in steps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps / steps_per_epoch, lrs, linewidth=2, color="#4CAF50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Decay Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def clear_session() -> None:
    """Release GPU/CPU memory between model training runs.

    Call this after saving results for one model and before building the next.
    It clears TensorFlow's Keras session and forces a Python garbage-collection
    pass, which typically reclaims several hundred MB in Colab.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    print("TF session cleared and garbage collected.")
