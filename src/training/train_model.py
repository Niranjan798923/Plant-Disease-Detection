import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt

from src.preprocessing.data_generator import create_data_generators

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 15
FINE_TUNE_AT = 100

def build_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model

def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

def get_callbacks():
    return [
        ModelCheckpoint(
            "models/best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger("outputs/metrics/training_log.csv")
    ]

def plot_history(history, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train():
    print("Loading data generators...")
    data = create_data_generators()
    num_classes = data['num_classes']
    class_names = data['class_names']
    
    print(f"Building model for {num_classes} classes...")
    model, base_model = build_model(num_classes)
    compile_model(model, INITIAL_LR)
    
    print(model.summary())
    
    print("\n=== Phase 1: Training head only ===")
    history1 = model.fit(
        data['train'],
        steps_per_epoch=data['steps_per_epoch'],
        validation_data=data['val'],
        validation_steps=data['val_steps'],
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    plot_history(history1, "Phase 1: Head Training", "outputs/plots/training_phase1.png")
    
    print("\n=== Phase 2: Fine-tuning ===")
    base_model.trainable = True
    
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    
    compile_model(model, FINE_TUNE_LR)
    
    history2 = model.fit(
        data['train'],
        steps_per_epoch=data['steps_per_epoch'],
        validation_data=data['val'],
        validation_steps=data['val_steps'],
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    plot_history(history2, "Phase 2: Fine-tuning", "outputs/plots/training_phase2.png")
    
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history.get(key, [])
    
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    plot_history(CombinedHistory(combined_history), "Complete Training", "outputs/plots/training_complete.png")
    
    model.save("models/final_model.keras")
    print("Model saved to models/final_model.keras")
    
    with open("outputs/metrics/training_history.json", "w") as f:
        json.dump(combined_history, f, indent=2)
    
    return model, data

if __name__ == "__main__":
    train()