import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Tomato Doctor AI — Plant Pathology Platform",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Disease Treatment & Care Knowledge Base
TREATMENT_GUIDES = {
    "Bacterial_spot": {
        "action": "Apply copper-based fungicides early.",
        "steps": [
            "Avoid overhead watering; water at the base of the plant.",
            "Prune infected foliage and sanitize tools with 70% alcohol.",
            "Practice a 2-3 year crop rotation with non-solanaceous crops."
        ]
    },
    "Early_blight": {
        "action": "Remove lower affected leaves and spray protective fungicide.",
        "steps": [
            "Apply fungicides containing chlorothalonil or copper octanoate.",
            "Mulch around the base to prevent soil spores from splashing onto leaves.",
            "Ensure 18–24 inches of spacing between plants for optimal airflow."
        ]
    },
    "Late_blight": {
        "action": "Urgent intervention required — high spread rate.",
        "steps": [
            "Apply systemic fungicides (e.g., Mancozeb or Copper hydroxides) immediately.",
            "Remove and destroy (do not compost) severely infected plants.",
            "Keep leaf foliage completely dry during cool, damp weather."
        ]
    },
    "Leaf_Mold": {
        "action": "Reduce ambient humidity and improve ventilation.",
        "steps": [
            "Increase greenhouse/garden ventilation and spacing.",
            "Avoid foliage wetness during evening and night hours.",
            "Apply bio-fungicides containing Bacillus subtilis."
        ]
    },
    "Septoria_leaf_spot": {
        "action": "Remove lower infected foliage promptly.",
        "steps": [
            "Pick off spotted leaves at first sign of infection.",
            "Apply organic copper fungicide every 7 to 10 days.",
            "Keep garden clean of end-of-season plant debris."
        ]
    },
    "Tomato_mosaic_virus": {
        "action": "No chemical cure — isolate and manage vectors.",
        "steps": [
            "Remove and burn or discard infected plants immediately.",
            "Wash hands and sanitize tools thoroughly after handling.",
            "Control aphid populations and avoid using tobacco products near plants."
        ]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "action": "Control whitefly vectors immediately.",
        "steps": [
            "Use insecticidal soaps, neem oil, or yellow sticky traps for whiteflies.",
            "Cover young seedlings with fine mesh insect netting.",
            "Remove weeds around the garden that act as alternate viral hosts."
        ]
    },
    "healthy": {
        "action": "Plant is thriving — maintain current care routine!",
        "steps": [
            "Ensure 6–8 hours of direct daily sunlight.",
            "Maintain consistent deep watering (1–1.5 inches per week).",
            "Apply balanced organic fertilizer every 3–4 weeks."
        ]
    }
}

# 3. Custom CSS & Visual Styling
st.markdown("""
    <style>
    /* Hide Default Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}

    /* Realistic Tomato Logo Floating Animation */
    .hero-container {
        text-align: center;
        padding-top: 10px;
        margin-bottom: 25px;
    }
    .tomato-img {
        width: 140px;
        height: auto;
        filter: drop-shadow(0px 12px 20px rgba(229, 57, 53, 0.35));
        transition: transform 0.4s ease;
        animation: float 4s ease-in-out infinite;
    }
    .tomato-img:hover {
        transform: scale(1.08) rotate(4deg);
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    /* Headings */
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #555555;
        font-size: 1.05rem;
        font-weight: 500;
    }

    /* Interactive Cards with Pop-Up Hover Effect */
    .card-container {
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
    }
    .card-container:hover {
        transform: translateY(-7px) scale(1.015);
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.12);
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
        font-weight: 800;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .healthy-text { color: #1b5e20; }
    .infected-text { color: #b71c1c; }

    .confidence-container {
        font-size: 1.05rem;
        font-weight: 600;
        color: #333333;
        margin-top: 6px;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #1976d2;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-left: 8px;
    }

    /* Treatment Guide Card */
    .treatment-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s ease;
    }
    .treatment-card:hover {
        transform: translateY(-4px);
    }
    .treatment-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 8px;
    }

    /* Uploader Hover Styling */
    div[data-testid="stFileUploader"] {
        border-radius: 14px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.12);
    }
    </style>
""", unsafe_allow_html=True)

# 4. Load Neural Network & Class Maps
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

# Sidebar Navigation & Information
with st.sidebar:
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/1202/1202125.png" width="90" style="filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));">
            <h2 style="margin-top:10px; color:#1b5e20;">Tomato Doctor AI</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("Automated deep learning platform for real-time tomato leaf disease detection.")
    st.markdown("---")
    st.markdown("### Supported Categories:")
    if class_indices:
        for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            clean_name = name.replace("Tomato___", "").replace("_", " ")
            st.markdown(f"- **{clean_name}**")
    st.markdown("---")
    st.caption("MobileNetV2 Neural Architecture")

# Hero Section with Realistic High-Res Graphic
st.markdown("""
    <div class="hero-container">
        <img src="https://cdn-icons-png.flaticon.com/512/1202/1202125.png" class="tomato-img" alt="Tomato Doctor Logo">
        <div class="main-title">Tomato Doctor AI</div>
        <div class="sub-title">Upload a leaf sample below for immediate AI-powered pathology analysis</div>
    </div>
""", unsafe_allow_html=True)

# Main Application Logic
if model is None:
    st.error("⚠️ Model files not detected! Please ensure `model/tomato_model.h5` and `model/class_indices.json` exist in your repository.")
else:
    uploaded_file = st.file_uploader("Choose a tomato leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("##### 📷 Uploaded Leaf Sample")
            img_display = Image.open(uploaded_file)
            st.image(img_display, use_column_width=True)
            
        with col2:
            st.markdown("##### 📋 Diagnostic Report")
            with st.spinner("Analyzing cellular patterns & leaf pathology..."):
                img = img_display.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                predictions = model.predict(img_array)
                pred_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]) * 100)
                
                raw_label = class_indices[str(pred_idx)]
                disease_key = raw_label.replace("Tomato___", "")
                disease_name = disease_key.replace("_", " ")
                is_healthy = "healthy" in raw_label.lower()

            # Unified Status Header + Confidence Below
            if is_healthy:
                st.markdown(f"""
                    <div class="card-container card-healthy">
                        <div class="result-header healthy-text">
                            🌿 Status: Healthy (No Disease Detected)
                        </div>
                        <div class="confidence-container">
                            Confidence Score: <b>{confidence:.2f}%</b>
                            <span class="confidence-badge">High Certainty</span>
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
                            Confidence Score: <b>{confidence:.2f}%</b>
                            <span class="confidence-badge">AI Flagged</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Actionable Treatment Guide Card
        guide_info = TREATMENT_GUIDES.get(disease_key, TREATMENT_GUIDES.get("healthy"))
        if guide_info:
            st.markdown(f"""
                <div class="treatment-card">
                    <div class="treatment-title">💡 Recommended Care & Treatment Plan</div>
                    <p style="margin-bottom: 10px; font-weight: 600; color: #1b5e20;">Primary Action: {guide_info['action']}</p>
                    <ul style="margin-bottom: 0; padding-left: 20px; color: #444;">
                        {''.join([f'<li style="margin-bottom: 4px;">{step}</li>' for step in guide_info['steps']])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Expandable Probability Distribution
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Detailed Class Probability Breakdown"):
            for idx, prob in enumerate(predictions[0]):
                cat_name = class_indices[str(idx)].replace("Tomato___", "").replace("_", " ")
                st.progress(float(prob), text=f"{cat_name}: {prob*100:.2f}%")
