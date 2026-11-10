import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image

# 1. Page Setup
st.set_page_config(
    page_title="Tomato Doctor AI",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Professional CSS Styling & Branding Removal
st.markdown("""
    <style>
    /* Hide default Streamlit headers, footers, menus, and deploy buttons */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    /* Clean Title Formatting */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1b5e20;
        text-align: center;
        margin-top: -30px;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        color: #555555;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    /* Pop-up Hover Effect Cards */
    .card-container {
        border-radius: 14px;
        padding: 22px 24px;
        margin-top: 10px;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .card-container:hover {
        transform: translateY(-6px) scale(1.015);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.12);
    }
    
    .card-healthy {
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
        border-left: 8px solid #2e7d32;
    }
    
    .card-infected {
        background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
        border-left: 8px solid #d32f2f;
    }
    
    .result-header {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .healthy-text { color: #1b5e20; }
    .infected-text { color: #b71c1c; }
    
    .confidence-container {
        font-size: 1.05rem;
        font-weight: 600;
        color: #424242;
        margin-top: 6px;
    }
    .confidence-score {
        color: #0d47a1;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    /* File Uploader Hover Animation */
    div[data-testid="stFileUploader"] {
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        border-radius: 12px;
    }
    div[data-testid="stFileUploader"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

# Load Model & Labels Safely
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

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/tomato.png", width=75)
    st.title("Tomato Doctor AI")
    st.markdown("Precision computer vision assistant for automated leaf health diagnosis.")
    st.markdown("---")
    st.markdown("### Supported Categories:")
    if class_indices:
        for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            clean_name = name.replace("Tomato___", "").replace("_", " ")
            st.markdown(f"- **{clean_name}**")

# Main Page Header
st.markdown('<div class="main-title">🍅 Tomato Doctor AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a tomato leaf photo to analyze plant health</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model files missing! Please ensure `model/tomato_model.h5` and `model/class_indices.json` exist in your repository.")
else:
    uploaded_file = st.file_uploader("Upload leaf sample (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("##### Uploaded Sample")
            img_display = Image.open(uploaded_file)
            st.image(img_display, use_column_width=True)
            
        with col2:
            st.markdown("##### Diagnostic Report")
            with st.spinner("Analyzing neural network features..."):
                img = img_display.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                predictions = model.predict(img_array)
                pred_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]) * 100)
                
                raw_label = class_indices[str(pred_idx)]
                disease_name = raw_label.replace("Tomato___", "").replace("_", " ")
                is_healthy = "healthy" in raw_label.lower()

            # Unified Disease Status Header + Confidence Below
            if is_healthy:
                st.markdown(f"""
                    <div class="card-container card-healthy">
                        <div class="result-header healthy-text">
                            🌿 Status: Healthy (No Disease Detected)
                        </div>
                        <div class="confidence-container">
                            Confidence Score: <span class="confidence-score">{confidence:.2f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="card-container card-infected">
                        <div class="result-header infected-text">
                            ⚠️ Status: Disease Detected — {disease_name}
                        </div>
                        <div class="confidence-container">
                            Confidence Score: <span class="confidence-score">{confidence:.2f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Detailed Probability Breakdown"):
            for idx, prob in enumerate(predictions[0]):
                cat_name = class_indices[str(idx)].replace("Tomato___", "").replace("_", " ")
                st.progress(float(prob), text=f"{cat_name}: {prob*100:.2f}%")
