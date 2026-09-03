import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.predictor import DiseasePredictor, format_class_name
from src.recommendation.knowledge_base import get_recommendation, format_recommendation

st.set_page_config(
    page_title="Plant Disease Detection & Pesticide Recommendation",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_predictor():
    return DiseasePredictor()

def main():
    st.title("🌿 Plant Disease Detection & Pesticide Recommendation System")
    st.markdown("**Upload a plant leaf image to detect diseases and get treatment recommendations**")
    
    predictor = load_predictor()
    
    with st.sidebar:
        st.header("About")
        st.info("""
        This system uses a MobileNetV2-based deep learning model 
        trained on the PlantVillage dataset to detect 13 classes 
        of plant diseases on pepper, potato, and tomato.
        
        **Model Performance:**
        - Test Accuracy: 92.57%
        - Test Precision: 94.23%
        - Test Recall: 91.51%
        """)
        
        st.header("Supported Classes")
        for cls in sorted(predictor.class_indices.keys()):
            st.write(f"• {format_class_name(cls)}")
    
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of a plant leaf"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
            temp_path = "temp_upload.jpg"
            image.save(temp_path)
        
        with col2:
            st.subheader("Prediction Results")
            
            with st.spinner("Analyzing image..."):
                try:
                    predictions = predictor.predict(temp_path, top_k=3)
                    
                    if predictions:
                        top_pred = predictions[0]
                        
                        confidence = top_pred['confidence']
                        if confidence >= 0.8:
                            st.success(f"**{format_class_name(top_pred['class'])}**")
                        elif confidence >= 0.5:
                            st.warning(f"**{format_class_name(top_pred['class'])}**")
                        else:
                            st.error(f"**{format_class_name(top_pred['class'])}**")
                        
                        st.metric("Confidence", top_pred['confidence_percent'])
                        
                        st.write("**Top 3 Predictions:**")
                        for i, pred in enumerate(predictions, 1):
                            st.write(f"{i}. {format_class_name(pred['class'])}: {pred['confidence_percent']}")
                        
                        st.divider()
                        
                        st.subheader("Treatment Recommendation")
                        rec = get_recommendation(top_pred['class'])
                        
                        if rec['pesticides']:
                            st.write("**Recommended Pesticides:**")
                            for p in rec['pesticides']:
                                with st.expander(f"{p['name']} ({p['type']})"):
                                    st.write(f"**Application:** {p['application']}")
                                    st.write(f"**Timing:** {p['timing']}")
                                    st.write(f"**Notes:** {p['notes']}")
                        else:
                            st.info("No specific pesticide treatment required for this condition.")
                        
                        st.write("**Cultural Practices:**")
                        for cp in rec['cultural_practices']:
                            st.write(f"• {cp}")
                        
                        st.write("**Preventive Measures:**")
                        for pm in rec['preventive_measures']:
                            st.write(f"• {pm}")
                        
                        st.caption("⚠️ Follow product labels and local agricultural authority guidance for dosage, application rates, and safety precautions.")
                    else:
                        st.error("Could not generate predictions.")
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    else:
        st.info("👆 Please upload a leaf image to get started")
        
        st.divider()
        
        st.subheader("Sample Test Images")
        st.write("You can test with images from the dataset:")
        
        sample_cols = st.columns(4)
        sample_classes = [
            ("Tomato_Late_blight", "data/PlantVillage/Tomato_Late_blight"),
            ("Potato_Early_blight", "data/PlantVillage/Potato___Early_blight"),
            ("Pepper_Bacterial_spot", "data/PlantVillage/Pepper__bell___Bacterial_spot"),
            ("Tomato_healthy", "data/PlantVillage/Pepper__bell___healthy")
        ]
        
        for idx, (label, path) in enumerate(sample_classes):
            with sample_cols[idx]:
                if os.path.exists(path):
                    files = [f for f in os.listdir(path) if f.endswith('.JPG') or f.endswith('.jpg')]
                    if files:
                        sample_img = Image.open(os.path.join(path, files[0]))
                        st.image(sample_img, caption=label, use_container_width=True)

if __name__ == "__main__":
    main()