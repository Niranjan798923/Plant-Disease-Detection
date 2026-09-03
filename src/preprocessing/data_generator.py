import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import json

DATASET_PATH = "data/PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def get_class_names():
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    return classes

def create_data_generators(validation_split=0.2, test_split=0.1):
    classes = get_class_names()
    num_classes = len(classes)
    
    all_image_paths = []
    all_labels = []
    
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(DATASET_PATH, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            all_image_paths.append(os.path.join(cls_path, img))
            all_labels.append(idx)
    
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths, all_labels, test_size=test_split, random_state=SEED, stratify=all_labels
    )
    
    val_size = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=SEED, stratify=y_train_val
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    def create_generator(paths, labels, datagen, shuffle=True):
        def generator():
            while True:
                indices = np.arange(len(paths))
                if shuffle:
                    np.random.shuffle(indices)
                for i in range(0, len(paths), BATCH_SIZE):
                    batch_indices = indices[i:i+BATCH_SIZE]
                    batch_paths = paths[batch_indices]
                    batch_labels = labels[batch_indices]
                    
                    batch_images = []
                    for p in batch_paths:
                        img = tf.keras.utils.load_img(p, target_size=IMG_SIZE)
                        img_array = tf.keras.utils.img_to_array(img)
                        batch_images.append(img_array)
                    
                    batch_images = np.array(batch_images)
                    batch_images = datagen.standardize(batch_images)
                    yield batch_images, tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)
        
        return generator()
    
    train_gen = create_generator(X_train, y_train, train_datagen, shuffle=True)
    val_gen = create_generator(X_val, y_val, val_test_datagen, shuffle=False)
    test_gen = create_generator(X_test, y_test, val_test_datagen, shuffle=False)
    
    steps_per_epoch = len(X_train) // BATCH_SIZE
    val_steps = len(X_val) // BATCH_SIZE
    test_steps = len(X_test) // BATCH_SIZE
    
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    
    with open("outputs/metrics/class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)
    
    return {
        'train': train_gen,
        'val': val_gen,
        'test': test_gen,
        'steps_per_epoch': steps_per_epoch,
        'val_steps': val_steps,
        'test_steps': test_steps,
        'num_classes': num_classes,
        'class_names': classes,
        'class_indices': class_indices,
        'X_test': X_test,
        'y_test': y_test
    }

def preprocess_single_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

if __name__ == "__main__":
    data = create_data_generators()
    print(f"Classes: {data['class_names']}")
    print(f"Num classes: {data['num_classes']}")
    print(f"Steps per epoch: {data['steps_per_epoch']}")
    print(f"Val steps: {data['val_steps']}")
    print(f"Test steps: {data['test_steps']}")