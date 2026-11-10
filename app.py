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

# 3. Custom CSS & Visual Animations
st.markdown("""
    <style>
    /* Hide Default Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}

    /* Keyframe Animations */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(1.5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes pulse-glow {
        0% { filter: drop-shadow(0 0 10px rgba(46, 125, 50, 0.25)); }
        50% { filter: drop-shadow(0 0 25px rgba(46, 125, 50, 0.5)); }
        100% { filter: drop-shadow(0 0 10px rgba(46, 125, 50, 0.25)); }
    }

    @keyframes pulse-badge {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .farmer-svg {
        animation: float 4.5s ease-in-out infinite, pulse-glow 3.5s ease-in-out infinite;
        transition: transform 0.4s ease;
    }
    .farmer-svg:hover {
        transform: scale(1.08) rotate(-2deg);
    }

    /* Headings */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 14px;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #555555;
        font-size: 1.05rem;
        font-weight: 500;
    }

    /* Result Cards */
    .card-container {
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
    }
    .card-container:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.14);
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
        background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-left: 8px;
        animation: pulse-badge 2.5s infinite ease-in-out;
    }

    /* Treatment Guide Card */
    .treatment-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 26px;
        margin-top: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .treatment-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    }
    .treatment-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 10px;
    }

    /* Image Preview Zoom Effect */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.03);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
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

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center;">
            <svg width="80" height="80" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="100" r="90" fill="#E8F5E9" stroke="#2E7D32" stroke-width="4"/>
                <path d="M40 90 Q100 70 160 90 Q170 95 160 100 Q100 115 40 100 Z" fill="#E5A93C"/>
                <path d="M65 90 Q70 45 100 45 Q130 45 135 90 Z" fill="#F3BF55"/>
                <ellipse cx="100" cy="115" rx="28" ry="24" fill="#FFCC80"/>
                <circle cx="90" cy="112" r="3" fill="#333"/>
                <circle cx="110" cy="112" r="3" fill="#333"/>
                <path d="M92 124 Q100 132 108 124" stroke="#333" stroke-width="2.5" fill="none"/>
                <path d="M60 170 Q100 145 140 170 L150 200 L50 200 Z" fill="#1565C0"/>
                <circle cx="132" cy="142" r="18" fill="#E53935"/>
                <path d="M132 124 L128 118 M132 124 L136 118" stroke="#4CAF50" stroke-width="3"/>
            </svg>
            <h2 style="margin-top:10px; color:#1b5e20;">Tomato Doctor AI</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("Automated deep learning platform for real-time tomato leaf disease diagnosis.")
    st.markdown("---")
    st.markdown("### Supported Categories:")
    if class_indices:
        for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            clean_name = name.replace("Tomato___", "").replace("_", " ")
            st.markdown(f"- **{clean_name}**")
    st.markdown("---")
    st.caption("MobileNetV2 Neural Network")

# Hero Section with Vector SVG Farmer Wearing Straw Hat Holding Tomato
st.markdown("""
    <div class="hero-container">
        <svg width="180" height="180" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" class="farmer-svg">
            <circle cx="100" cy="100" r="90" fill="url(#bg-grad)" stroke="#2E7D32" stroke-width="4"/>
            <path d="M35 95 Q100 72 165 95 Q175 100 165 105 Q100 120 35 105 Q25 100 35 95 Z" fill="#E5A93C" stroke="#B37B14" stroke-width="2"/>
            <path d="M62 93 Q68 45 100 45 Q132 45 138 93 Z" fill="#F3BF55" stroke="#B37B14" stroke-width="2"/>
            <path d="M62 88 Q100 80 138 88 Q137 94 137 94 Q100 85 63 94 Z" fill="#C62828"/>
            <ellipse cx="100" cy="118" rx="28" ry="24" fill="#FFCC80"/>
            <circle cx="89" cy="114" r="3.5" fill="#212121"/>
            <circle cx="111" cy="114" r="3.5" fill="#212121"/>
            <path d="M91 126 Q100 135 109 126" stroke="#212121" stroke-width="3" stroke-linecap="round" fill="none"/>
            <path d="M55 172 Q100 148 145 172 L155 200 L45 200 Z" fill="#1565C0"/>
            <path d="M80 160 L80 200 M120 160 L120 200" stroke="#FFD54F" stroke-width="4"/>
            <circle cx="136" cy="144" r="19" fill="#E53935" stroke="#B71C1C" stroke-width="2"/>
            <path d="M136 125 L131 118 M136 125 L141 118 M136 125 L136 116" stroke="#4CAF50" stroke-width="3.5" stroke-linecap="round"/>
            <path d="M118 148 Q124 158 132 155" stroke="#FFCC80" stroke-width="4.5" stroke-linecap="round" fill="none"/>
            <defs>
                <linearGradient id="bg-grad" x1="0" y1="0" x2="200" y2="200">
                    <stop offset="0%" stop-color="#E8F5E9"/>
                    <stop offset="100%" stop-color="#C8E6C9"/>
                </linearGradient>
            </defs>
        </svg>
        <div class="main-title">Tomato Doctor AI</div>
        <div class="sub-title">Provide a leaf sample below using your camera or file upload for instant AI analysis</div>
    </div>
""", unsafe_allow_html=True)

# Main Application Logic
if model is None:
    st.error("⚠️ Model files not detected! Please ensure `model/tomato_model.h5` and `model/class_indices.json` exist in your repository.")
else:
    # 5. Dual Input Option: File Upload vs Camera Capture
    input_type = st.radio(
        "Choose Image Input Method:",
        ["📁 Upload Image File", "📷 Take Photo with Camera"],
        horizontal=True
    )

    uploaded_file = None
    if input_type == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Choose a tomato leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Click photo of the tomato leaf")

    if uploaded_file is not None:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("##### 📷 Leaf Sample")
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

            # Result Header + Confidence Score
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
