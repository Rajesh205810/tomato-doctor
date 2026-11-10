import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Tomato Doctor - High Accuracy Detector",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# UI Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        color: #1b5e20;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-header {
        text-align: center;
        color: #4f5b66;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .result-healthy {
        padding: 20px;
        border-radius: 12px;
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        margin-top: 15px;
    }
    .result-infected {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffebee;
        border-left: 6px solid #c62828;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
    model_path = "model/tomato_model.h5"
    labels_path = "model/class_indices.json"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None
        
    loaded_model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        class_indices = json.load(f)
    return loaded_model, class_indices

model, class_indices = load_model_and_labels()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/tomato.png", width=70)
    st.title("Tomato Doctor AI")
    st.markdown("Powered by EfficientNet and fine-tuned on your tomato dataset for high-precision disease diagnostics.")
    st.markdown("---")
    st.markdown("### Detectable Conditions:")
    if class_indices:
        for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            clean_name = name.replace("Tomato___", "").replace("_", " ")
            st.markdown(f"- {clean_name}")
    st.markdown("---")
    st.caption("Ready for Streamlit Cloud deployment via GitHub.")

# Main Header
st.markdown('<p class="main-header">🍅 Tomato Doctor AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload or test any similar tomato leaf image to instantly identify health status and infections.</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model files not detected! Please ensure `model/tomato_model.h5` and `model/class_indices.json` are committed to your GitHub repository.")
else:
    uploaded_file = st.file_uploader("Upload a tomato leaf image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### Uploaded Sample")
            img_display = Image.open(uploaded_file)
            st.image(img_display, use_column_width=True)
            
        with col2:
            st.markdown("#### Diagnostic Result")
            with st.spinner("Analyzing patterns and scanning leaf features..."):
                # Preprocess image for EfficientNetB0
                img = img_display.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                # Predict
                predictions = model.predict(img_array)
                pred_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]) * 100)
                
                raw_label = class_indices[str(pred_idx)]
                disease_name = raw_label.replace("Tomato___", "").replace("_", " ")
                is_healthy = "healthy" in raw_label.lower()
                
            # Display formatted card
            if is_healthy:
                st.markdown(f"""
                    <div class="result-healthy">
                        <h3 style="color: #2e7d32; margin-top:0;">Status: Healthy 🌿</h3>
                        <p><b>Condition:</b> {disease_name}</p>
                        <p><b>Confidence Score:</b> <b>{confidence:.2f}%</b></p>
                        <p style="margin-bottom:0; font-size:0.9rem;">No signs of pathology detected on this leaf.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-infected">
                        <h3 style="color: #c62828; margin-top:0;">Status: Infected ⚠️</h3>
                        <p><b>Detected Disease:</b> {disease_name}</p>
                        <p><b>Confidence Score:</b> <b>{confidence:.2f}%</b></p>
                        <p style="margin-bottom:0; font-size:0.9rem;">Recommended: Review phytosanitary steps for treatment.</p>
                    </div>
                """, unsafe_allow_html=True)

        # Full Probability Breakdown
        with st.expander("📊 View Detailed Confidence Across All Classes"):
            for idx, prob in enumerate(predictions[0]):
                cat_name = class_indices[str(idx)].replace("Tomato___", "").replace("_", " ")
                st.progress(float(prob), text=f"{cat_name}: {prob*100:.2f}%")