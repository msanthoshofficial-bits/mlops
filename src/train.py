import mlflow
import mlflow.tensorflow
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/server environments
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.preprocess import get_train_val_generators
from src.model import create_model
import tensorflow as tf


def log_training_curves(history):
    """Generate training curves and log directly to MLflow (no disk files)."""
    epochs_range = range(1, len(history.history['loss']) + 1)

    # --- Loss Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss')
    ax.plot(epochs_range, history.history['val_loss'], 'r-o', label='Validation Loss')
    ax.set_title('Training & Validation Loss', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    mlflow.log_figure(fig, "charts/loss_curve.png")
    plt.close(fig)

    # --- Accuracy Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy')
    ax.plot(epochs_range, history.history['val_accuracy'], 'r-o', label='Validation Accuracy')
    ax.set_title('Training & Validation Accuracy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    mlflow.log_figure(fig, "charts/accuracy_curve.png")
    plt.close(fig)


def log_confusion_matrix(model, val_gen):
    """Generate confusion matrix and classification report, log directly to MLflow."""
    # Reset generator and collect predictions
    val_gen.reset()
    y_true = val_gen.classes
    steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    y_pred_probs = model.predict(val_gen, steps=steps)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Trim to match (in case of rounding in steps)
    y_true = y_true[:len(y_pred)]

    class_names = list(val_gen.class_indices.keys())

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    fig.tight_layout()
    mlflow.log_figure(fig, "charts/confusion_matrix.png")
    plt.close(fig)

    # --- Classification Report (logged as text) ---
    report = classification_report(y_true, y_pred, target_names=class_names)
    mlflow.log_text(report, "charts/classification_report.txt")

    # --- Log individual class metrics ---
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    for class_name in class_names:
        mlflow.log_metric(f"{class_name}_precision", report_dict[class_name]['precision'])
        mlflow.log_metric(f"{class_name}_recall", report_dict[class_name]['recall'])
        mlflow.log_metric(f"{class_name}_f1", report_dict[class_name]['f1-score'])


def train_model():
    # Use local dataset
    base_dir = r"C:\dev\mlops\Dataset\PetImages"
    
    if not os.path.exists(base_dir):
        print(f"Error: Dataset not found at {base_dir}")
        return

    print(f"Training on data from: {base_dir}")
    train_gen, val_gen = get_train_val_generators(base_dir)

    # MLflow Tracking
    mlflow.set_experiment("cats_vs_dogs")
    
    with mlflow.start_run():
        # Parameters
        img_size = (224, 224)
        batch_size = 32
        epochs = 2
        learning_rate = 1e-4

        mlflow.log_param("img_size", img_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("base_model", "MobileNetV2")

        # Create Model
        model = create_model(input_shape=img_size + (3,))
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.1),
            tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy')
        ]

        # Train
        history = model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // batch_size,
            callbacks=callbacks
        )

        # ── Log Metrics (per-epoch for MLflow charts) ────────────
        for epoch_idx in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch_idx], step=epoch_idx + 1)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch_idx], step=epoch_idx + 1)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch_idx], step=epoch_idx + 1)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch_idx], step=epoch_idx + 1)

        # Final metrics
        mlflow.log_metric("final_loss", history.history['loss'][-1])
        mlflow.log_metric("final_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

        # ── Log Visualizations directly to MLflow ────────────────
        print("Logging training curves to MLflow...")
        log_training_curves(history)

        print("Logging confusion matrix to MLflow...")
        log_confusion_matrix(model, val_gen)

        # ── Save Model ───────────────────────────────────────────
        if os.path.exists("best_model.h5"):
            if os.path.exists("model.h5"):
                os.remove("model.h5")
            os.rename("best_model.h5", "model.h5")
            
        mlflow.log_artifact("model.h5")
        
        print("=" * 60)
        print("Training complete!")
        print(f"  Final Accuracy:     {history.history['accuracy'][-1]:.4f}")
        print(f"  Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Model saved to:     model.h5")
        print(f"  View MLflow UI:     python -m mlflow ui")
        print(f"  All charts viewable at: http://localhost:5000")
        print("=" * 60)


if __name__ == "__main__":
    train_model()
