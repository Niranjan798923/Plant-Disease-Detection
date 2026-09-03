import os
import json
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd

DATASET_PATH = "data/PlantVillage"
OUTPUT_DIR = "outputs/metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_dataset():
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    print(f"Found {len(classes)} classes:")
    class_counts = {}
    class_images = {}
    
    for cls in classes:
        cls_path = os.path.join(DATASET_PATH, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[cls] = len(images)
        class_images[cls] = images
        print(f"  {cls}: {len(images)} images")
    
    total_images = sum(class_counts.values())
    print(f"\nTotal images: {total_images}")
    print(f"Average per class: {total_images / len(classes):.1f}")
    print(f"Min class size: {min(class_counts.values())} ({min(class_counts, key=class_counts.get)})")
    print(f"Max class size: {max(class_counts.values())} ({max(class_counts, key=class_counts.get)})")
    
    sample_class = classes[0]
    sample_img_path = os.path.join(DATASET_PATH, sample_class, class_images[sample_class][0])
    with Image.open(sample_img_path) as img:
        print(f"\nSample image size: {img.size}")
        print(f"Sample image mode: {img.mode}")
    
    stats = {
        "num_classes": len(classes),
        "class_names": classes,
        "class_counts": class_counts,
        "total_images": total_images,
        "avg_per_class": total_images / len(classes),
        "min_class_size": min(class_counts.values()),
        "max_class_size": max(class_counts.values()),
        "sample_image_size": list(Image.open(sample_img_path).size),
        "sample_image_mode": Image.open(sample_img_path).mode
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
    
    print(f"\nDataset stats saved to {OUTPUT_DIR}/dataset_stats.json")
    print(f"Class distribution saved to {OUTPUT_DIR}/class_distribution.csv")
    
    return classes, class_counts

if __name__ == "__main__":
    analyze_dataset()