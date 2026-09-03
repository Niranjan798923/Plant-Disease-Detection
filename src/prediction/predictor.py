import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

MODEL_PATH = "models/best_model.keras"
CLASS_INDICES_PATH = "outputs/metrics/class_indices.json"

class DiseasePredictor:
    def __init__(self, model_path=MODEL_PATH, class_indices_path=CLASS_INDICES_PATH):
        self.model = tf.keras.models.load_model(model_path)
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        self.num_classes = len(self.class_indices)
        self.img_size = (224, 224)
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict(self, image_path, top_k=3):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        results = []
        for idx in top_indices:
            class_name = self.idx_to_class[idx]
            confidence = float(predictions[idx])
            results.append({
                'class': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            })
        return results
    
    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            try:
                pred = self.predict(path)
                results.append({
                    'image': path,
                    'predictions': pred,
                    'top_prediction': pred[0] if pred else None,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'image': path,
                    'predictions': [],
                    'top_prediction': None,
                    'error': str(e)
                })
        return results

def format_class_name(class_name):
    name = class_name.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
    return name

if __name__ == "__main__":
    predictor = DiseasePredictor()
    print(f"Loaded model with {predictor.num_classes} classes")
    print("Classes:", list(predictor.class_indices.keys()))
    
    test_image = "data/PlantVillage/Pepper__bell___Bacterial_spot/0a6d8c3f-5c1b-4c3e-8b4d-5f8e1a2b3c4d___Pepper__bell___Bacterial_spot.JPG"
    if os.path.exists(test_image):
        results = predictor.predict(test_image)
        print(f"\nTest prediction for {test_image}:")
        for r in results:
            print(f"  {format_class_name(r['class'])}: {r['confidence_percent']}")
    else:
        print("Test image not found, skipping test prediction")