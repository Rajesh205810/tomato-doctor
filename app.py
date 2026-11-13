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
    page_title="Plant Doctor — AI Leaf Diagnostic",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize Session State for User Account & Language
if "app_lang" not in st.session_state:
    st.session_state.app_lang = "English"
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

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

# 2. Bilingual Dictionary (English & Nepali)
TEXT = {
    "English": {
        "brand_name": "Plant Doctor",
        "brand_sub": "AI-Powered Tomato Leaf Disease Detection",
        "how_title": "✨ How Does It Work?",
        "step1_title": "1. Capture or Upload",
        "step1_desc": "Take a clear photo of the tomato leaf or upload an image from your device.",
        "step2_title": "2. AI Deep Scan",
        "step2_desc": "Our neural network inspects leaf spots, texture, and color patterns.",
        "step3_title": "3. Instant Result",
        "step3_desc": "View the exact detected disease and its confidence score immediately.",
        "tab_upload": "📁 Upload Photo",
        "tab_camera": "📷 Take Photo",
        "upload_label": "Choose a tomato leaf image (JPG, JPEG, PNG)...",
        "camera_label": "Snap a live photo of your tomato leaf",
        "sample_header": "📷 Leaf Sample",
        "report_header": "📋 Diagnostic Report",
        "spinner_text": "Analyzing leaf patterns...",
        "status_healthy": "🌿 Status: Healthy (No Disease Detected)",
        "status_infected": "⚠️ Disease Detected:",
        "confidence_label": "Confidence Score:",
        "badge_healthy": "High Certainty",
        "badge_infected": "AI Flagged",
        "prob_expander": "📊 Detailed Class Probability Breakdown",
        # Sidebar Translations
        "account_header": "👤 Gmail Account",
        "name_placeholder": "Your Full Name",
        "email_placeholder": "yourname@gmail.com",
        "login_btn": "Sign in with Gmail",
        "login_error": "Please enter a valid @gmail.com address and your name.",
        "login_success": "Account created & signed in!",
        "scans_label": "Leaves Scanned:",
        "logout_btn": "Sign Out",
        "settings_header": "⚙️ Settings",
        "lang_label": "App Language / भाषा",
        "ai_header": "✨ Further Research with AI",
        "ai_desc": "Want deeper insights about plant care or crop science? Continue your research on Google Gemini.",
        "ai_btn": "🚀 Open Google Gemini",
        "supported_header": "🌿 Detectable Diseases",
        "model_missing": "⚠️ Model files not detected! Ensure `model/tomato_model.h5` and `model/class_indices.json` are in your repository.",
        "diseases": {
            "Bacterial_spot": "Bacterial Spot",
            "Early_blight": "Early Blight",
            "Late_blight": "Late Blight",
            "Leaf_Mold": "Leaf Mold",
            "Septoria_leaf_spot": "Septoria Leaf Spot",
            "Tomato_mosaic_virus": "Tomato Mosaic Virus",
            "Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
            "healthy": "Healthy Plant"
        }
    },
    "नेपाली": {
        "brand_name": "प्लान्ट डाक्टर (Plant Doctor)",
        "brand_sub": "एआई-संचालित गोलभेडाको पातको रोग पहिचान प्रणाली",
        "how_title": "✨ यसले कसरी काम गर्छ?",
        "step1_title": "१. फोटो खिच्नुहोस् वा अपलोड गर्नुहोस्",
        "step1_desc": "गोलभेडाको पातको स्पष्ट फोटो खिच्नुहोस् वा ग्यालरीबाट छान्नुहोस्।",
        "step2_title": "२. एआई गहिरो परीक्षण",
        "step2_desc": "हाम्रो एआई प्रणालीले पातको दाग, बनावट र रङ परिवर्तनको विश्लेषण गर्छ।",
        "step3_title": "३. तत्काल नतिजा",
        "step3_desc": "पहिचान भएको रोगको नाम र विश्वसनीयता प्रतिशत (Confidence Score) तुरुन्तै हेर्नुहोस्।",
        "tab_upload": "📁 फोटो अपलोड गर्नुहोस्",
        "tab_camera": "📷 फोटो खिच्नुहोस्",
        "upload_label": "गोलभेडाको पातको फोटो छान्नुहोस् (JPG, JPEG, PNG)...",
        "camera_label": "गोलभेडाको पातको प्रत्यक्ष फोटो खिच्नुहोस्",
        "sample_header": "📷 पातको नमुना",
        "report_header": "📋 रोग परीक्षण प्रतिवेदन",
        "spinner_text": "पातको लक्षण विश्लेषण गर्दै...",
        "status_healthy": "🌿 अवस्था: स्वस्थ (कुनै रोग भेटिएन)",
        "status_infected": "⚠️ रोग पहिचान भयो:",
        "confidence_label": "विश्वसनीयता स्तर (Confidence Score):",
        "badge_healthy": "उच्च निश्चितता",
        "badge_infected": "एआईद्वारा पहिचान",
        "prob_expander": "📊 विस्तृत सम्भावना विवरण",
        # Sidebar Translations
        "account_header": "👤 जिमेल खाता (Gmail Account)",
        "name_placeholder": "तपाईंको पूरा नाम",
        "email_placeholder": "yourname@gmail.com",
        "login_btn": "जिमेलबाट लगइन गर्नुहोस्",
        "login_error": "कृपया मान्य @gmail.com ठेगाना र आफ्नो नाम लेख्नुहोस्।",
        "login_success": "खाता सिर्जना र लगइन सफल भयो!",
        "scans_label": "परीक्षण गरिएका पातहरू:",
        "logout_btn": "लगआउट (Sign Out)",
        "settings_header": "⚙️ सेटिङहरू (Settings)",
        "lang_label": "भाषा छान्नुहोस् / Language",
        "ai_header": "✨ एआईसँग थप अनुसन्धान",
        "ai_desc": "बिरुवाको बारेमा थप विस्तृत जानकारी खोज्न गुगल जेमिनी (Google Gemini) प्रयोग गर्नुहोस्।",
        "ai_btn": "🚀 गुगल जेमिनी खोल्नुहोस्",
        "supported_header": "🌿 पहिचान हुने रोगहरू",
        "model_missing": "⚠️ मोडेल फाइलहरू भेटिएनन्! कृपया `model/tomato_model.h5` र `model/class_indices.json` जाँच गर्नुहोस्।",
        "diseases": {
            "Bacterial_spot": "ब्याक्टेरियल स्पट (Bacterial Spot)",
            "Early_blight": "अर्ली ब्लाइट / डढुवा (Early Blight)",
            "Late_blight": "लेट ब्लाइट / पछौटे डढुवा (Late Blight)",
            "Leaf_Mold": "लिफ मोल्ड / पातको ढुसी (Leaf Mold)",
            "Septoria_leaf_spot": "सेप्टोरिया लिफ स्पट (Septoria Leaf Spot)",
            "Tomato_mosaic_virus": "टोमाटो मोजाइक भाइरस (Mosaic Virus)",
            "Tomato_Yellow_Leaf_Curl_Virus": "पात बेरिने पहेँलो भाइरस (Yellow Leaf Curl)",
            "healthy": "स्वस्थ बिरुवा (Healthy)"
        }
    }
}

# 3. Custom CSS & Sliding Sidebar Styling
st.markdown("""
    <style>
    /* Hide Default Streamlit Branding while KEEPING the Sidebar Toggle Button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Prominent Green Sidebar Open/Close Toggle Button */
    [data-testid="collapsedControl"], button[kind="headerNoPadding"] {
        background-color: #2e7d32 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        margin: 8px !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.35) !important;
        transition: transform 0.2s ease !important;
    }
    [data-testid="collapsedControl"]:hover, button[kind="headerNoPadding"]:hover {
        transform: scale(1.08) !important;
        background-color: #1b5e20 !important;
    }

    /* Smooth Sliding Animation for Sidebar */
    section[data-testid="stSidebar"] {
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), width 0.4s ease !important;
    }
    .sidebar-slide {
        animation: slideInLeft 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes slideInLeft {
        0% { transform: translateX(-30px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    /* User Profile Card inside Sidebar */
    .profile-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%);
        border: 1.5px solid #81c784;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.1);
    }
    .profile-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: #2e7d32;
        color: white;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* Bright Curved Rectangle Header Box */
    .bright-header-box {
        background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 55%, #f1f8e9 100%);
        border: 2px solid #81c784;
        border-radius: 26px;
        padding: 18px 28px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        box-shadow: 0 12px 32px rgba(76, 175, 80, 0.18), inset 0 0 24px rgba(255, 255, 255, 0.95);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .bright-header-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 38px rgba(76, 175, 80, 0.26), inset 0 0 24px rgba(255, 255, 255, 1);
    }

    /* Ultra-Sharp Compact Circular Logo */
    .logo-circle {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        background: #ffffff;
        border: 3px solid #1b5e20;
        padding: 2px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.25);
        transform: translateZ(0);
    }
    .logo-circle img {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        object-position: center 15%;
        image-rendering: -webkit-optimize-contrast;
        image-rendering: high-quality;
        backface-visibility: hidden;
        transform: translateZ(0);
    }

    .header-text-group {
        text-align: left;
    }
    .header-brand-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #1b5e20;
        line-height: 1.15;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-brand-sub {
        font-size: 0.98rem;
        font-weight: 600;
        color: #2e7d32;
        margin-top: 4px;
    }

    /* Bright High-Visibility "How Does It Work?" Section */
    .how-title-wrapper {
        text-align: center;
        margin-bottom: 14px;
    }
    .how-section-title {
        display: inline-block;
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: #ffffff !important;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 7px 22px;
        border-radius: 999px;
        letter-spacing: 0.2px;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.28);
    }
    .step-card {
        background: linear-gradient(180deg, #ffffff 0%, #f1f8e9 100%);
        border: 2px solid #a5d6a7;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        height: 100%;
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.08);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }
    .step-card:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: #2e7d32;
        background: #ffffff;
        box-shadow: 0 12px 26px rgba(46, 125, 50, 0.18);
    }
    .step-title {
        font-weight: 800;
        font-size: 1.0rem;
        color: #1b5e20;
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: 0.88rem;
        font-weight: 500;
        color: #334155;
        line-height: 1.4;
    }

    /* Side-by-Side Tabs */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        border-radius: 12px 12px 0 0 !important;
    }

    /* Diagnostic Result Cards */
    .card-container {
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
    }
    .card-container:hover {
        transform: translateY(-7px) scale(1.02);
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.14);
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
        margin-bottom: 8px;
        line-height: 1.35;
    }
    .healthy-text { color: #1b5e20; }
    .infected-text { color: #b71c1c; }

    .confidence-container {
        font-size: 1.08rem;
        font-weight: 600;
        color: #212121;
        margin-top: 8px;
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
    }
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

# 5. Sliding Sidebar: Gmail Account, Language Settings & Gemini AI Research
with st.sidebar:
    st.markdown('<div class="sidebar-slide">', unsafe_allow_html=True)
    t = TEXT[st.session_state.app_lang]

    # --- SECTION 1: GMAIL ACCOUNT LOGIN / PROFILE ---
    st.markdown(f"### {t['account_header']}")
    if st.session_state.user_email is None:
        with st.form("gmail_login_form", clear_on_submit=False):
            input_name = st.text_input("Name", placeholder=t["name_placeholder"], label_visibility="collapsed")
            input_email = st.text_input("Gmail", placeholder=t["email_placeholder"], label_visibility="collapsed")
            submitted = st.form_submit_button(f"🔐 {t['login_btn']}", use_container_width=True)
            
            if submitted:
                clean_email = input_email.strip().lower()
                clean_name = input_name.strip()
                if clean_name and clean_email.endswith("@gmail.com") and len(clean_email) > 10:
                    st.session_state.user_name = clean_name
                    st.session_state.user_email = clean_email
                    st.success(t["login_success"])
                    st.rerun()
                else:
                    st.error(t["login_error"])
    else:
        initial = st.session_state.user_name[0].upper()
        st.markdown(f"""
            <div class="profile-card">
                <div style="display:flex; align-items:center;">
                    <div class="profile-avatar">{initial}</div>
                    <div>
                        <div style="font-weight:800; color:#1b5e20; font-size:1rem;">{st.session_state.user_name}</div>
                        <div style="font-size:0.8rem; color:#475569;">{st.session_state.user_email}</div>
                    </div>
                </div>
                <div style="margin-top:10px; font-size:0.85rem; font-weight:600; color:#2e7d32;">
                    📊 {t['scans_label']} {st.session_state.scan_count}
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(f"🚪 {t['logout_btn']}", use_container_width=True):
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.rerun()

    st.markdown("---")

    # --- SECTION 2: SETTINGS (LANGUAGE SWITCHER) ---
    st.markdown(f"### {t['settings_header']}")
    selected_lang = st.radio(
        t["lang_label"],
        options=["English", "नेपाली"],
        index=0 if st.session_state.app_lang == "English" else 1,
        horizontal=True
    )
    if selected_lang != st.session_state.app_lang:
        st.session_state.app_lang = selected_lang
        st.rerun()

    st.markdown("---")

    # --- SECTION 3: FURTHER RESEARCH WITH GEMINI AI ---
    st.markdown(f"### {t['ai_header']}")
    st.caption(t["ai_desc"])
    st.link_button(t["ai_btn"], "https://gemini.google.com", use_container_width=True)

    st.markdown("---")

    # --- SECTION 4: SUPPORTED CATEGORIES ---
    with st.expander(t["supported_header"]):
        if class_indices:
            for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
                raw_key = name.replace("Tomato___", "")
                translated_name = t["diseases"].get(raw_key, raw_key.replace("_", " "))
                st.markdown(f"- **{translated_name}**")

    st.markdown('</div>', unsafe_allow_html=True)

# Refresh active translation dictionary
t = TEXT[st.session_state.app_lang]

# 6. Bright Curved Header Box (Sharp Circular Farmer Logo + "Plant Doctor")
logo_html = f'<img src="{farmer_b64}" alt="Plant Doctor Logo">' if farmer_b64 else '<span style="font-size:2rem;">🌱</span>'

st.markdown(f"""
    <div class="bright-header-box">
        <div class="logo-circle">
            {logo_html}
        </div>
        <div class="header-text-group">
            <div class="header-brand-title">{t['brand_name']}</div>
            <div class="header-brand-sub">{t['brand_sub']}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 7. Bright Interactive "How Does It Work?" Section
st.markdown(f'<div class="how-title-wrapper"><div class="how-section-title">{t["how_title"]}</div></div>', unsafe_allow_html=True)
h_col1, h_col2, h_col3 = st.columns(3, gap="medium")

with h_col1:
    st.markdown(f"""
        <div class="step-card">
            <div class="step-title">{t['step1_title']}</div>
            <div class="step-desc">{t['step1_desc']}</div>
        </div>
    """, unsafe_allow_html=True)

with h_col2:
    st.markdown(f"""
        <div class="step-card">
            <div class="step-title">{t['step2_title']}</div>
            <div class="step-desc">{t['step2_desc']}</div>
        </div>
    """, unsafe_allow_html=True)

with h_col3:
    st.markdown(f"""
        <div class="step-card">
            <div class="step-title">{t['step3_title']}</div>
            <div class="step-desc">{t['step3_desc']}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 8. Main Image Input (Upload Photo or Take Photo) & Diagnosis
if model is None:
    st.error(t["model_missing"])
else:
    tab1, tab2 = st.tabs([t["tab_upload"], t["tab_camera"]])

    uploaded_file = None
    with tab1:
        file_input = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"], key="uploader")
        if file_input is not None:
            uploaded_file = file_input

    with tab2:
        camera_input = st.camera_input(t["camera_label"], key="camera")
        if camera_input is not None:
            uploaded_file = camera_input

    if uploaded_file is not None:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown(f"##### {t['sample_header']}")
            img_display = Image.open(uploaded_file).convert("RGB")
            st.image(img_display, use_container_width=True)
            
        with col2:
            st.markdown(f"##### {t['report_header']}")
            with st.spinner(t["spinner_text"]):
                img = img_display.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                predictions = model.predict(img_array)
                pred_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]) * 100)
                
                raw_label = class_indices[str(pred_idx)]
                disease_key = raw_label.replace("Tomato___", "")
                disease_name = t["diseases"].get(disease_key, disease_key.replace("_", " "))
                is_healthy = "healthy" in raw_label.lower()
                st.session_state.scan_count += 1

            if is_healthy:
                st.markdown(f"""
                    <div class="card-container card-healthy">
                        <div class="result-header healthy-text">
                            {t['status_healthy']}
                        </div>
                        <div class="confidence-container">
                            {t['confidence_label']} <b>{confidence:.2f}%</b>
                            <span class="confidence-badge">{t['badge_healthy']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="card-container card-infected">
                        <div class="result-header infected-text">
                            {t['status_infected']} {disease_name}
                        </div>
                        <div class="confidence-container">
                            {t['confidence_label']} <b>{confidence:.2f}%</b>
                            <span class="confidence-badge">{t['badge_infected']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Expandable Probability Distribution
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(t["prob_expander"]):
            for idx, prob in enumerate(predictions[0]):
                raw_k = class_indices[str(idx)].replace("Tomato___", "")
                cat_name = t["diseases"].get(raw_k, raw_k.replace("_", " "))
                st.progress(float(prob), text=f"{cat_name}: {prob*100:.2f}%")
