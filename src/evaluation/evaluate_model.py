import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing.data_generator import create_data_generators, preprocess_single_image

MODEL_PATH = "models/best_model.keras"
OUTPUT_DIR = "outputs/metrics"
PLOTS_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_model_and_data():
    model = tf.keras.models.load_model(MODEL_PATH)
    data = create_data_generators()
    return model, data

def evaluate_on_test(model, data):
    class_names = data['class_names']
    X_test = data['X_test']
    y_test = data['y_test']
    test_steps = data['test_steps']
    
    print(f"Evaluating on {len(X_test)} test images...")
    
    y_pred_probs = model.predict(data['test'], steps=test_steps, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test[:len(y_pred)]
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        data['test'], steps=test_steps, verbose=1
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Loss:      {test_loss:.4f}")
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_loss": float(test_loss),
            "classification_report": report
        }, f, indent=2)
    
    plot_confusion_matrix(cm, class_names)
    plot_per_class_metrics(report, class_names)
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'loss': test_loss,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_probs': y_pred_probs.tolist()
    }

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(14, 12))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_raw.png"), dpi=150)
    plt.close()

def plot_per_class_metrics(report, class_names):
    classes = [c for c in class_names if c in report]
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall, and F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_class_metrics.png"), dpi=150)
    plt.close()
    
    df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)

def plot_training_curves():
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    if not os.path.exists(log_path):
        return
    
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(df['epoch'], df['accuracy'], label='Train Accuracy', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df['epoch'], df['loss'], label='Train Loss', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(df['epoch'], df['precision'], label='Train Precision', marker='o')
    axes[1, 0].plot(df['epoch'], df['val_precision'], label='Val Precision', marker='s')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(df['epoch'], df['recall'], label='Train Recall', marker='o')
    axes[1, 1].plot(df['epoch'], df['val_recall'], label='Val Recall', marker='s')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle('Training Curves (Best Model Checkpoint)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    print("Loading model and data...")
    model, data = load_model_and_data()
    
    print("Running evaluation...")
    results = evaluate_on_test(model, data)
    
    print("Plotting training curves...")
    plot_training_curves()
    
    print(f"\nEvaluation complete. Results saved to {OUTPUT_DIR}/")
    print(f"Plots saved to {PLOTS_DIR}/")