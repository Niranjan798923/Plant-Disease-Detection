# Plant Disease Detection and Pesticide Recommendation System
## CIA 3 Component 1 - Project Documentation

**Course:** CSEDS743E04 – Image and Video Analytics  
**Program:** B.Tech – Artificial Intelligence and Machine Learning & Data Science  
**Date:** September 2026  

---

## SECTION 1: PROJECT OBJECTIVES

### 1.1 Problem Statement
Agriculture is a critical sector for food security and economic productivity. Plant diseases caused by fungi, bacteria, viruses, and pests significantly reduce crop yield, quality, and farmer income. Traditional diagnosis depends on manual visual inspection, which is time-consuming, subjective, and requires expert knowledge often unavailable in rural areas.

### 1.2 Motivation
- Automate plant disease diagnosis using computer vision and deep learning
- Provide actionable treatment recommendations to farmers
- Reduce dependence on manual expert inspection
- Minimize unnecessary pesticide usage through precise diagnosis
- Support precision and sustainable agriculture

### 1.3 Overall Objective
Develop an end-to-end automated system that analyzes plant leaf images, identifies diseases using deep learning, and provides intelligent pesticide/treatment recommendations.

### 1.4 Specific Objectives
1. **Automated Disease Detection**: Develop a system that analyzes plant leaf images using image processing and computer vision to identify healthy and diseased plants
2. **Deep Learning Classification**: Implement a CNN-based model capable of recognizing multiple plant diseases by learning disease-specific visual features
3. **Intelligent Recommendation Module**: Design a knowledge-based system providing suitable pesticides, application guidance, and preventive measures based on detected disease
4. **Precision Agriculture**: Promote early diagnosis, reduce manual inspection dependence, minimize pesticide misuse, and support informed decision-making

### 1.5 Expected Contribution
- Integrated disease detection + treatment recommendation (addressing research gap)
- Practical tool for farmers and agricultural extension workers
- Demonstration of transfer learning for agricultural computer vision
- Structured pesticide recommendation knowledge base

---

## SECTION 2: MODULE DESIGN

### 2.1 Overall Architecture
```
USER → LEAF IMAGE INPUT → IMAGE PREPROCESSING → DISEASE CLASSIFIER → PREDICTED DISEASE
                                                           ↓
                                              +------------+------------+
                                              ↓                         ↓
                                       CONFIDENCE           RECOMMENDATION MODULE
                                                              ↓
                                                TREATMENT/PESTICIDE INFO
                                                              ↓
                                                PREVENTIVE GUIDANCE
                                                              ↓
                                                       USER OUTPUT
```

### 2.2 Module Responsibilities

#### Module 1: Image Input (`src/prediction/predictor.py`)
- Accept leaf image files (JPG, PNG)
- Validate file type and readability
- Handle invalid inputs gracefully

#### Module 2: Image Preprocessing (`src/preprocessing/data_generator.py`)
- Resize images to 224×224 (MobileNetV2 input size)
- Normalize using MobileNetV2 preprocessing (preprocess_input)
- Data augmentation during training (rotation, shift, zoom, flip)
- Train/Validation/Test split (70%/20%/10% stratified)

#### Module 3: Disease Detection (`src/training/train_model.py`, `src/prediction/predictor.py`)
- **Model**: MobileNetV2 with transfer learning (ImageNet weights)
- **Architecture**: Base model (frozen initially) → GlobalAveragePooling2D → BatchNorm → Dropout(0.3) → Dense(512, relu) → BatchNorm → Dropout(0.3) → Dense(13, softmax)
- **Training Phases**: 
  - Phase 1: Train head only (30 epochs, LR=1e-4)
  - Phase 2: Fine-tune top layers (15 epochs, LR=1e-5, unfreeze after layer 100)
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
- **Output**: Class prediction + confidence scores

#### Module 4: Recommendation System (`src/recommendation/knowledge_base.py`)
- Structured knowledge base mapping 13 disease classes to treatments
- Each entry contains: disease info, pesticides (with type, application, timing, notes), cultural practices, preventive measures
- Handles unknown diseases with default safe recommendation
- Safety disclaimer: "Follow product label and local agricultural authority guidance"

#### Module 5: User Interface (`app/streamlit_app.py`)
- Streamlit web application
- Image upload with preview
- Real-time prediction with confidence display
- Treatment recommendations with expandable details
- Sample test images for demonstration

#### Module 6: Model Evaluation (`src/evaluation/evaluate_model.py`)
- Test set evaluation (accuracy, precision, recall, F1)
- Classification report (per-class metrics)
- Confusion matrix (normalized and raw)
- Training curves (accuracy, loss, precision, recall)
- Per-class performance visualization

### 2.3 Data Flow
1. User uploads leaf image via Streamlit UI
2. Image preprocessed (resize 224×224, MobileNetV2 normalization)
3. Model predicts disease class + confidence scores
4. Top prediction passed to recommendation module
5. Knowledge base returns structured treatment info
6. Results displayed in UI with formatted recommendations

---

## SECTION 3: IMPLEMENTATION

### 3.1 Development Environment
- **Language**: Python 3.13
- **Framework**: TensorFlow 2.15+, Keras
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, OpenCV, Pillow, Streamlit
- **Hardware**: CPU training (no GPU on Windows TensorFlow 2.11+)
- **Dataset Source**: PlantVillage via Kaggle (emmarex/plantdisease)

### 3.2 Dataset
- **Name**: PlantVillage (subset)
- **Classes**: 13 classes (3 crops: Pepper, Potato, Tomato)
- **Total Images**: 17,195
- **Class Distribution**:
  - Pepper__bell___Bacterial_spot: 997
  - Pepper__bell___healthy: 1,478
  - Potato___Early_blight: 1,000
  - Potato___Late_blight: 1,000
  - Potato___healthy: 152
  - Tomato_Bacterial_spot: 2,127
  - Tomato_Early_blight: 1,000
  - Tomato_Late_blight: 1,909
  - Tomato_Leaf_Mold: 952
  - Tomato_Septoria_leaf_spot: 1,771
  - Tomato_Spider_mites_Two_spotted_spider_mite: 1,676
  - Tomato__Target_Spot: 1,404
  - Tomato__Tomato_YellowLeaf__Curl_Virus: 1,729
- **Image Size**: 256×256 RGB (resized to 224×224 for training)
- **Split**: Train 12,036 / Val 3,439 / Test 1,720 (stratified)

### 3.3 Preprocessing
- **Training Augmentation**: Rotation(20°), Width/Height shift(0.2), Shear(0.2), Zoom(0.2), Horizontal flip
- **Validation/Test**: Only normalization (preprocess_input)
- **Normalization**: MobileNetV2 preprocessing (scales to [-1, 1])

### 3.4 Model Selection: MobileNetV2
**Rationale**:
- Lightweight (3.5M parameters) - suitable for deployment
- Good accuracy-efficiency tradeoff
- Pre-trained on ImageNet (transfer learning effective)
- Fast inference for real-time demo
- 13-class classification achieved >92% test accuracy

### 3.5 Training Configuration
| Parameter | Phase 1 (Head) | Phase 2 (Fine-tune) |
|-----------|---------------|---------------------|
| Epochs | 30 | 15 |
| Learning Rate | 1e-4 | 1e-5 |
| Optimizer | Adam | Adam |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Batch Size | 32 | 32 |
| Trainable Layers | Head only | Layers 100+ unfrozen |
| Early Stopping | Patience=7 | Patience=7 |

### 3.6 Recommendation Knowledge Base
- 13 disease-specific entries + 1 default
- Each entry: disease name, pathogen, description, pesticides (2-7 per disease), cultural practices (4-6), preventive measures (4-6)
- Pesticide info: name, type/class, application method, timing, resistance management notes
- Sources: Standard agricultural extension recommendations (general knowledge)

### 3.7 UI Implementation
- Streamlit single-page application
- Sidebar: model info, class list
- Main: upload widget, image preview, predictions, recommendations
- Error handling for invalid inputs
- Sample images for quick testing

---

## SECTION 4: RESULTS

### 4.1 Dataset Statistics
- **Total Images**: 17,195 across 13 classes
- **Class Imbalance**: Present (152 to 2,127 images per class)
- **Mitigation**: Stratified splitting, class weights not used (sufficient samples per class)

### 4.2 Model Performance (Test Set - 1,720 images)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.57%** |
| **Test Precision** | **94.23%** |
| **Test Recall** | **91.51%** |
| **Test F1-Score (Weighted)** | **92.47%** |
| **Test Loss** | 0.2158 |

### 4.3 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Pepper__bell___Bacterial_spot | 0.98 | 0.99 | 0.99 | 100 |
| Pepper__bell___healthy | 0.97 | 1.00 | 0.99 | 146 |
| Potato___Early_blight | 0.96 | 0.96 | 0.96 | 99 |
| Potato___Late_blight | 0.92 | 0.95 | 0.94 | 100 |
| Potato___healthy | 1.00 | 0.71 | 0.83 | 14 |
| Tomato_Bacterial_spot | 0.92 | 0.97 | 0.94 | 209 |
| Tomato_Early_blight | 0.84 | 0.72 | 0.77 | 99 |
| Tomato_Late_blight | 0.97 | 0.93 | 0.95 | 186 |
| Tomato_Leaf_Mold | 0.93 | 0.87 | 0.90 | 94 |
| Tomato_Septoria_leaf_spot | 0.88 | 0.84 | 0.86 | 176 |
| Tomato_Spider_mites | 0.91 | 0.96 | 0.94 | 167 |
| Tomato__Target_Spot | 0.85 | 0.93 | 0.89 | 139 |
| Tomato_YellowLeaf_Curl_Virus | 0.96 | 0.96 | 0.96 | 167 |

### 4.4 Key Observations
- **Best Performing**: Pepper classes (>98% F1), Potato Early Blight (96%), Tomato Late Blight (95%), TYLCV (96%)
- **Challenging Classes**: Tomato Early Blight (77% F1), Potato Healthy (83% F1 - only 14 test samples), Tomato Septoria (86% F1)
- **Confusion**: Tomato Early Blight sometimes confused with Target Spot and Septoria (similar visual symptoms)

### 4.5 Training History
- **Phase 1 (Head Training)**: 9 epochs until early stopping (val_acc plateau at ~0.9216)
- **Phase 2 (Fine-tuning)**: Interrupted at epoch 10; best model checkpoint saved at epoch 8 (val_acc=0.9216)
- **Convergence**: Stable, no severe overfitting (train/val gap ~1-2%)

### 4.6 Generated Outputs
- **Models**: `models/best_model.keras` (17.6 MB)
- **Metrics**: `outputs/metrics/test_results.json`, `outputs/metrics/per_class_metrics.csv`
- **Plots**: 
  - `outputs/plots/confusion_matrix.png` (normalized)
  - `outputs/plots/confusion_matrix_raw.png`
  - `outputs/plots/per_class_metrics.png`
  - `outputs/plots/training_curves.png`

### 4.7 Sample Predictions (End-to-End Test)
| Input Image | Predicted Class | Confidence | Recommendation Generated |
|-------------|----------------|------------|-------------------------|
| Pepper Bacterial Spot | Pepper__bell___Bacterial_spot | 99.99% | ✓ 2 pesticides |
| Tomato Late Blight | Tomato_Late_blight | 99.71% | ✓ 5 pesticides |
| Potato Early Blight | Potato___Early_blight | 99.59% | ✓ 4 pesticides |
| Healthy Pepper | Pepper__bell___healthy | 100.00% | ✓ No pesticides (healthy) |

### 4.8 UI Demonstration
- Streamlit app launches at `http://localhost:8501`
- Real-time prediction (<2 seconds per image)
- Confidence-based color coding (green ≥80%, yellow 50-80%, red <50%)
- Expandable pesticide details with application guidance

---

## SECTION 5: DISCUSSION

### 5.1 Achievements
✅ All four CIA 1 objectives met:
1. Automated disease detection via deep learning (92.57% accuracy)
2. Multi-class classification (13 diseases across 3 crops)
3. Integrated pesticide recommendation knowledge base
4. End-to-end working demonstration system

### 5.2 Limitations
1. **Dataset Bias**: PlantVillage images are lab-controlled; real-world performance may vary
2. **Class Imbalance**: Potato Healthy only 152 samples (71% recall)
3. **No Field Validation**: Tested only on dataset images
4. **CPU Only**: Training slower without GPU
5. **Pesticide Data**: General recommendations; region-specific registrations not included
6. **Viral Diseases**: TYLCV managed via vector control (whitefly), not direct antiviral treatment

### 5.3 Future Scope
- Field data collection and validation
- Mobile app deployment (TensorFlow Lite)
- Expansion to more crops and diseases
- Integration with weather/disease forecasting APIs
- Region-specific pesticide registration database
- Segmentation for lesion localization

---

## SECTION 6: PROJECT STRUCTURE

```
cia3/
├── data/
│   └── PlantVillage/          # 13 class folders, 17,195 images
├── models/
│   └── best_model.keras       # Trained MobileNetV2 (17.6 MB)
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── dataset_analysis.py
│   │   └── data_generator.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_model.py
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_model.py
│   └── recommendation/
│       ├── __init__.py
│       └── knowledge_base.py
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── outputs/
│   ├── metrics/
│   │   ├── dataset_stats.json
│   │   ├── class_distribution.csv
│   │   ├── class_indices.json
│   │   ├── training_log.csv
│   │   ├── test_results.json
│   │   └── per_class_metrics.csv
│   └── plots/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_raw.png
│       ├── per_class_metrics.png
│       └── training_curves.png
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

---

## SECTION 7: HOW TO RUN

### 7.1 Installation
```bash
pip install -r requirements.txt
```

### 7.2 Training (if needed)
```bash
$env:PYTHONPATH="."; python src/training/train_model.py
```

### 7.3 Evaluation
```bash
$env:PYTHONPATH="."; python src/evaluation/evaluate_model.py
```

### 7.4 Run Application
```bash
$env:PYTHONPATH="."; streamlit run app/streamlit_app.py
```
Then open `http://localhost:8501`

### 7.5 Test Prediction
```bash
$env:PYTHONPATH="."; python test_predictor.py
```

---

## SECTION 8: ACADEMIC INTEGRITY

- **Original Work**: All code written for this project
- **Dataset**: PlantVillage (public domain, cited)
- **Model Architecture**: MobileNetV2 (Google, cited via TensorFlow Keras Applications)
- **Pesticide Recommendations**: Based on standard agricultural extension knowledge; no proprietary sources
- **No Fabricated Results**: All metrics from actual executed experiments

---

*End of CIA 3 Component 1 Documentation*