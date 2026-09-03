# Plant Disease Detection and Pesticide Recommendation System

**Course:** CSEDS743E04 – Image and Video Analytics  
**Program:** B.Tech – Artificial Intelligence and Machine Learning & Data Science  
**Project:** CIA 3 Mini Project  

---

## Overview

An end-to-end automated system for plant disease detection using deep learning (MobileNetV2 transfer learning) with integrated intelligent pesticide recommendation. The system analyzes plant leaf images, identifies diseases across 13 classes (pepper, potato, tomato), and provides structured treatment recommendations including pesticides, cultural practices, and preventive measures.

### Key Features
- **Deep Learning Classification**: MobileNetV2-based model with 92.57% test accuracy
- **13 Disease Classes**: Covers bacterial, fungal, viral diseases and pests + healthy classes
- **Pesticide Recommendations**: Structured knowledge base with application guidance
- **Web Interface**: Streamlit-based UI for real-time prediction and recommendation
- **Complete Pipeline**: Image input → Preprocessing → Classification → Recommendation → Output

---

## Dataset

**PlantVillage** (via Kaggle: `emmarex/plantdisease`)

| Class | Images |
|-------|--------|
| Pepper__bell___Bacterial_spot | 997 |
| Pepper__bell___healthy | 1,478 |
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 152 |
| Tomato_Bacterial_spot | 2,127 |
| Tomato_Early_blight | 1,000 |
| Tomato_Late_blight | 1,909 |
| Tomato_Leaf_Mold | 952 |
| Tomato_Septoria_leaf_spot | 1,771 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 1,676 |
| Tomato__Target_Spot | 1,404 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 1,729 |
| **Total** | **17,195** |

Split: Train 12,036 / Validation 3,439 / Test 1,720 (stratified)

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **92.57%** |
| **Test Precision** | **94.23%** |
| **Test Recall** | **91.51%** |
| **Test F1 (Weighted)** | **92.47%** |

### Per-Class F1 Scores
- Pepper Bacterial Spot: 99%
- Pepper Healthy: 99%
- Potato Early Blight: 96%
- Potato Late Blight: 94%
- Tomato Bacterial Spot: 94%
- Tomato Late Blight: 95%
- Tomato Spider Mites: 94%
- Tomato Yellow Leaf Curl Virus: 96%
- Tomato Leaf Mold: 90%
- Tomato Target Spot: 89%
- Tomato Septoria Leaf Spot: 86%
- Potato Healthy: 83% (limited samples)
- Tomato Early Blight: 77%

---

## Project Structure

```
cia3/
├── data/
│   └── PlantVillage/              # Dataset (13 classes)
├── models/
│   └── best_model.keras           # Trained model (17.6 MB)
├── src/
│   ├── preprocessing/             # Data analysis & generators
│   ├── training/                  # Model training script
│   ├── prediction/                # Inference module
│   ├── evaluation/                # Metrics & visualization
│   └── recommendation/            # Pesticide knowledge base
├── app/
│   └── streamlit_app.py           # Web UI
├── outputs/
│   ├── metrics/                   # JSON/CSV results
│   └── plots/                     # Confusion matrix, curves
├── docs/
│   └── CIA3_Component1_Documentation.md
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- TensorFlow 2.15+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn, OpenCV, Pillow, Streamlit

---

## Usage

### 1. Run the Web Application
```bash
$env:PYTHONPATH="."; streamlit run app/streamlit_app.py
```
Opens at `http://localhost:8501`

### 2. Train Model (if needed)
```bash
$env:PYTHONPATH="."; python src/training/train_model.py
```

### 3. Evaluate Model
```bash
$env:PYTHONPATH="."; python src/evaluation/evaluate_model.py
```

### 4. Test Prediction
```bash
$env:PYTHONPATH="."; python test_predictor.py
```

---

## System Architecture

```
User Upload
    ↓
Image Preprocessing (224×224, MobileNetV2 normalization)
    ↓
MobileNetV2 Classifier (Transfer Learning)
    ↓
Predicted Disease + Confidence
    ↓
┌──────────────────┬──────────────────┐
│   Confidence     │ Recommendation   │
│   Display        │ Knowledge Base   │
└──────────────────┴──────────────────┘
    ↓
Treatment/Pesticide Info + Cultural Practices + Prevention
    ↓
User Output (Streamlit UI)
```

---

## Recommendation System

The knowledge base (`src/recommendation/knowledge_base.py`) provides for each disease:
- **Disease Info**: Name, pathogen, description
- **Pesticides**: Name, type/class, application method, timing, resistance notes
- **Cultural Practices**: Crop rotation, sanitation, irrigation management
- **Preventive Measures**: Resistant varieties, monitoring, seed treatment

**Safety Disclaimer**: "Follow the product label and local agricultural authority guidance for dosage, application rates, pre-harvest intervals (PHI), and safety precautions."

---

## Results & Visualizations

Generated in `outputs/`:
- **Confusion Matrix** (normalized & raw): `outputs/plots/confusion_matrix.png`
- **Per-Class Metrics**: `outputs/plots/per_class_metrics.png`
- **Training Curves**: `outputs/plots/training_curves.png`
- **Test Results**: `outputs/metrics/test_results.json`
- **Per-Class CSV**: `outputs/metrics/per_class_metrics.csv`

---

## Limitations

1. **Lab Dataset**: PlantVillage images are controlled; real-world performance may vary
2. **Class Imbalance**: Potato Healthy only 152 samples (71% recall)
3. **CPU Training**: No GPU acceleration on Windows TensorFlow 2.11+
4. **No Field Validation**: Tested only on held-out dataset images
5. **General Pesticide Data**: Region-specific registrations not included

---

## Future Work

- Field data collection and real-world validation
- Mobile deployment (TensorFlow Lite / TensorFlow.js)
- Expansion to more crops and diseases
- Integration with weather/disease forecasting
- Region-specific pesticide registration database
- Lesion segmentation and severity estimation

---

## Academic Integrity

- All code developed for this project
- PlantVillage dataset: Hughes & Salathé (2015), public domain
- MobileNetV2: Sandler et al. (2018), via TensorFlow Keras Applications
- Pesticide recommendations: Standard agricultural extension knowledge
- **No fabricated metrics** - all results from actual experiments

---

## License

Academic project for educational purposes. See individual licenses for datasets and models used.

---

*CIA 3 Component 1 - Completed September 2026*