import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import base64
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Tomato Doctor AI — Plant Pathology Platform",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Base64 Image Loader for Local Assets
def get_image_base64(file_path):
    if os.path.exists(file_path):
        ext = file_path.split('.')[-1].lower()
        mime_type = "image/png" if ext == "png" else "image/jpeg"
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:{mime_type};base64,{encoded}"
    return None

farmer_b64 = get_image_base64("farmer.png")
tomato_b64 = get_image_base64("tomato.jpg")

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
st.markdown(f"""
    <style>
    /* Hide Default Streamlit Chrome */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none !important;}}
    div[data-testid="stDecoration"] {{display: none !important;}}
    div[data-testid="stToolbar"] {{display: none !important;}}

    /* Sliding Animation for Sidebar Content */
    section[data-testid="stSidebar"] {{
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}
    .sidebar-animated-content {{
        animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    @keyframes slideInLeft {{
        0% {{ transform: translateX(-40px); opacity: 0; }}
        100% {{ transform: translateX(0); opacity: 1; }}
    }}

    /* Keyframe Floating Animation for Small Logo */
    @keyframes float {{
        0% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-5px) rotate(1deg); }}
        100% {{ transform: translateY(0px) rotate(0deg); }}
    }}

    /* Hero Section & Compact Logo Styling */
    .hero-container {{
        text-align: center;
        padding: 5px 0 15px 0;
    }}
    .farmer-hero-img {{
        max-width: 85px;
        height: auto;
        border-radius: 50%;
        animation: float 4.5s ease-in-out infinite;
        transition: transform 0.3s ease, filter 0.3s ease;
        filter: drop-shadow(0 6px 14px rgba(46, 125, 50, 0.25));
    }}
    .farmer-hero-img:hover {{
        transform: scale(1.1) rotate(-3deg);
        filter: drop-shadow(0 10px 20px rgba(46, 125, 50, 0.4));
    }}

    /* Title Formatting */
    .main-title {{
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 8px;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }}
    .sub-title {{
        color: #555555;
        font-size: 1.05rem;
        font-weight: 500;
    }}

    /* Side-by-Side Modern Tabs */
    button[data-baseweb="tab"] {{
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        border-radius: 12px 12px 0 0 !important;
        transition: all 0.3s ease !important;
    }}

    /* Diagnostic Card Pop-Up Hover Effects */
    .card-container {{
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
    }}
    .card-container:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.14);
    }}

    .card-healthy {{
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
        border-left: 8px solid #2e7d32;
    }}
    .card-infected {{
        background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
        border-left: 8px solid #d32f2f;
    }}

    .result-header {{
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 6px;
        line-height: 1.3;
    }}
    .healthy-text {{ color: #1b5e20; }}
    .infected-text {{ color: #b71c1c; }}

    .confidence-container {{
        font-size: 1.05rem;
        font-weight: 600;
        color: #333333;
        margin-top: 6px;
    }}
    .confidence-badge {{
        display: inline-block;
        background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-left: 8px;
    }}

    /* Care & Treatment Guide Card */
    .treatment-card {{
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 26px;
        margin-top: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .treatment-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    }}
    .treatment-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 10px;
    }}

    /* Preview Image Zoom Effect */
    div[data-testid="stImage"] img {{
        border-radius: 12px;
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }}
    div[data-testid="stImage"] img:hover {{
        transform: scale(1.03);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }}
    </style>
""", unsafe_allow_html=True)

# 4. Load Neural Network & Class Mapping
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

# Sidebar Navigation with Sliding Content Animation
with st.sidebar:
    st.markdown('<div class="sidebar-animated-content">', unsafe_allow_html=True)
    if tomato_b64:
        st.markdown(f'''
            <div style="text-align: center;">
                <img src="{tomato_b64}" width="70" style="border-radius: 50%; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));">
                <h2 style="margin-top:8px; color:#1b5e20;">Tomato Doctor AI</h2>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="text-align:center; color:#1b5e20;">🍅 Tomato Doctor AI</h2>', unsafe_allow_html=True)
        
    st.markdown("Automated deep learning platform for real-time tomato leaf disease diagnosis.")
    st.markdown("---")
    st.markdown("### Supported Categories:")
    if class_indices:
        for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            clean_name = name.replace("Tomato___", "").replace("_", " ")
            st.markdown(f"- **{clean_name}**")
    st.markdown("---")
    st.caption("MobileNetV2 Neural Network")
    st.markdown('</div>', unsafe_allow_html=True)

# Hero Section
st.markdown('<div class="hero-container">', unsafe_allow_html=True)
if farmer_b64:
    st.markdown(f'<img src="{farmer_b64}" class="farmer-hero-img" alt="Farmer Logo">', unsafe_allow_html=True)
else:
    st.markdown('🍅', unsafe_allow_html=True)

st.markdown("""
        <div class="main-title">Tomato Doctor AI</div>
        <div class="sub-title">Provide a leaf sample below using your file manager or camera for instant AI analysis</div>
    </div>
""", unsafe_allow_html=True)

# Main Application Logic
if model is None:
    st.error("⚠️ Model files not detected! Please ensure `model/tomato_model.h5` and `model/class_indices.json` exist in your repository.")
else:
    # 5. Side-by-Side Input Tabs for Upload & Camera
    tab1, tab2 = st.tabs(["📁 Upload Image File", "📷 Take Photo with Camera"])

    uploaded_file = None
    with tab1:
        file_input = st.file_uploader("Choose a tomato leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"], key="uploader")
        if file_input is not None:
            uploaded_file = file_input

    with tab2:
        camera_input = st.camera_input("Snap a live photo of your tomato leaf", key="camera")
        if camera_input is not None:
            uploaded_file = camera_input

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
