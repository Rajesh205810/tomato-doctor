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

# 2. Complete Bilingual Dictionary (English & Nepali)
TEXT = {
    "English": {
        "brand_name": "Plant Doctor",
        "brand_sub": "AI-Powered Tomato Leaf Disease Diagnostic System",
        "how_title": "✨ How Does It Work?",
        "step1_title": "1. Capture or Upload",
        "step1_desc": "Take a clear photo of the tomato leaf or upload an image from your device.",
        "step2_title": "2. AI Deep Scan",
        "step2_desc": "Our neural network inspects leaf spots, texture, and color changes.",
        "step3_title": "3. Instant Cure Plan",
        "step3_desc": "Receive accurate disease detection and step-by-step treatment steps.",
        "tab_upload": "📁 Upload Photo",
        "tab_camera": "📷 Take Photo",
        "upload_label": "Choose a tomato leaf image (JPG, JPEG, PNG)...",
        "camera_label": "Snap a live photo of your tomato leaf",
        "sample_header": "📷 Leaf Sample",
        "report_header": "📋 Diagnostic Report",
        "spinner_text": "Analyzing cellular patterns & leaf pathology...",
        "status_healthy": "🌿 Status: Healthy (No Disease Detected)",
        "status_infected": "⚠️ Status: Disease Detected —",
        "confidence_label": "Confidence Score:",
        "badge_healthy": "High Certainty",
        "badge_infected": "AI Flagged",
        "treatment_header": "💡 Recommended Care & Treatment Plan",
        "primary_action": "Primary Action:",
        "prob_expander": "📊 Detailed Class Probability Breakdown",
        "settings_header": "⚙️ Settings",
        "lang_select": "Select Language / भाषा छान्नुहोस्",
        "ai_slider_header": "🤖 Talk with AI",
        "ai_slider_label": "Slide to connect with a free AI assistant:",
        "slider_off": "Off",
        "ai_prompt_text": "Need more agricultural advice? Chat directly with",
        "ai_btn_text": "🚀 Open AI Chat",
        "supported_header": "Supported Categories:",
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
        },
        "guides": {
            "Bacterial_spot": {
                "action": "Apply copper-based fungicides early.",
                "steps": [
                    "Avoid overhead watering; water directly at the base of the plant.",
                    "Prune infected foliage and sanitize tools with 70% alcohol.",
                    "Practice a 2–3 year crop rotation with non-solanaceous crops."
                ]
            },
            "Early_blight": {
                "action": "Remove lower affected leaves and spray protective fungicide.",
                "steps": [
                    "Apply fungicides containing chlorothalonil or copper octanoate.",
                    "Mulch around the base to prevent soil spores from splashing onto leaves.",
                    "Ensure 18–24 inches of spacing between plants for proper airflow."
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
                    "Increase greenhouse or garden ventilation and plant spacing.",
                    "Avoid wetting leaves during evening and night hours.",
                    "Apply bio-fungicides containing Bacillus subtilis."
                ]
            },
            "Septoria_leaf_spot": {
                "action": "Remove lower infected foliage promptly.",
                "steps": [
                    "Pick off spotted leaves at the very first sign of infection.",
                    "Apply organic copper fungicide every 7 to 10 days.",
                    "Keep the garden clean of end-of-season plant debris."
                ]
            },
            "Tomato_mosaic_virus": {
                "action": "No chemical cure — isolate and manage vectors.",
                "steps": [
                    "Remove and discard infected plants immediately.",
                    "Wash hands and sanitize tools thoroughly after handling.",
                    "Control aphid populations and avoid tobacco use near plants."
                ]
            },
            "Tomato_Yellow_Leaf_Curl_Virus": {
                "action": "Control whitefly vectors immediately.",
                "steps": [
                    "Use insecticidal soaps, neem oil, or yellow sticky traps for whiteflies.",
                    "Cover young seedlings with fine mesh insect netting.",
                    "Remove weeds around the garden that act as viral hosts."
                ]
            },
            "healthy": {
                "action": "Plant is thriving — maintain your current care routine!",
                "steps": [
                    "Ensure 6–8 hours of direct daily sunlight.",
                    "Maintain consistent deep watering (1–1.5 inches per week).",
                    "Apply balanced organic fertilizer every 3–4 weeks."
                ]
            }
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
        "step3_title": "३. तत्काल उपचार योजना",
        "step3_desc": "रोगको सही पहिचान र चरणबद्ध उपचार सल्लाह प्राप्त गर्नुहोस्।",
        "tab_upload": "📁 फोटो अपलोड गर्नुहोस्",
        "tab_camera": "📷 फोटो खिच्नुहोस्",
        "upload_label": "गोलभेडाको पातको फोटो छान्नुहोस् (JPG, JPEG, PNG)...",
        "camera_label": "गोलभेडाको पातको प्रत्यक्ष फोटो खिच्नुहोस्",
        "sample_header": "📷 पातको नमुना",
        "report_header": "📋 रोग परीक्षण प्रतिवेदन",
        "spinner_text": "पातको कोषिका र रोगको लक्षण विश्लेषण गर्दै...",
        "status_healthy": "🌿 अवस्था: स्वस्थ (कुनै रोग भेटिएन)",
        "status_infected": "⚠️ अवस्था: रोग पहिचान भयो —",
        "confidence_label": "विश्वसनीयता स्तर:",
        "badge_healthy": "उच्च निश्चितता",
        "badge_infected": "एआईद्वारा पहिचान",
        "treatment_header": "💡 सिफारिस गरिएको हेरचाह र उपचार योजना",
        "primary_action": "मुख्य कदम:",
        "prob_expander": "📊 विस्तृत सम्भावना विवरण",
        "settings_header": "⚙️ सेटिङहरू (Settings)",
        "lang_select": "भाषा छान्नुहोस् / Language",
        "ai_slider_header": "🤖 एआई सहायकसँग कुरा गर्नुहोस्",
        "ai_slider_label": "निःशुल्क एआईसँग जोडिन स्लाइडर सार्नुहोस्:",
        "slider_off": "बन्द (Off)",
        "ai_prompt_text": "थप कृषि सल्लाह चाहिन्छ? सिधै कुरा गर्नुहोस्:",
        "ai_btn_text": "🚀 एआई च्याट खोल्नुहोस्",
        "supported_header": "पहिचान हुने रोगहरू:",
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
        },
        "guides": {
            "Bacterial_spot": {
                "action": "सुरुवाती चरणमै कपरयुक्त ढुसीनाशक (Copper fungicide) छर्नुहोस्।",
                "steps": [
                    "बिरुवाको माथिबाट पानी नहाल्नुहोस्; सिधै जरामा मात्र पानी दिनुहोस्।",
                    "रोग लागेका पातहरू काटेर हटाउनुहोस् र औजारहरू सफा राख्नुहोस्।",
                    "हरेक २–३ वर्षमा बाली चक्र (Crop rotation) अपनाउनुहोस्।"
                ]
            },
            "Early_blight": {
                "action": "तल्लो भागका संक्रमित पातहरू हटाउनुहोस् र ढुसीनाशक छर्नुहोस्।",
                "steps": [
                    "क्लोरोथालोनिल (Chlorothalonil) वा कपरयुक्त औषधि प्रयोग गर्नुहोस्।",
                    "माटोबाट रोग सर्न नदिन बोटको फेदमा छापो (Mulch) हाल्नुहोस्।",
                    "हावा खेल्नको लागि बिरुवाहरू बीच १८–२४ इन्चको दूरी राख्नुहोस्।"
                ]
            },
            "Late_blight": {
                "action": "तत्काल उपचार आवश्यक — यो रोग छिटो फैलिन्छ।",
                "steps": [
                    "तुरुन्तै म्यानकोजेब (Mancozeb) वा कपर हाइड्रोक्साइड छर्नुहोस्।",
                    "धेरै संक्रमित बोटहरू उखेलेर नष्ट गर्नुहोस् (कम्पोस्ट मल नबनाउनुहोस्)।",
                    "चिसो र ओसिलो मौसममा पातहरू सुख्खा राख्ने प्रयास गर्नुहोस्।"
                ]
            },
            "Leaf_Mold": {
                "action": "हावाको ओसिलोपन घटाउनुहोस् र हावा आवतजावत बढाउनुहोस्।",
                "steps": [
                    "हरितगृह (Greenhouse) वा बारीमा हावा खेल्ने व्यवस्था मिलाउनुहोस्।",
                    "साँझ र रातको समयमा पात भिजाउने गरी पानी नहाल्नुहोस्।",
                    "जैविक ढुसीनाशक (Bacillus subtilis) प्रयोग गर्नुहोस्।"
                ]
            },
            "Septoria_leaf_spot": {
                "action": "संक्रमित तल्ला पातहरू तुरुन्तै टिपेर हटाउनुहोस्।",
                "steps": [
                    "थोप्ला देखिने बित्तिकै ती पातहरू हटाएर नष्ट गर्नुहोस्।",
                    "हरेक ७ देखि १० दिनमा अर्गानिक कपर ढुसीनाशक छर्नुहोस्।",
                    "खेती सकिएपछि बारीका पुराना बोटबिरुवा सफा गर्नुहोस्।"
                ]
            },
            "Tomato_mosaic_virus": {
                "action": "यसको रासायनिक उपचार छैन — संक्रमित बोट अलग गर्नुहोस्।",
                "steps": [
                    "रोग लागेका बिरुवाहरू तुरुन्तै उखेलेर हटाउनुहोस्।",
                    "बिरुवा छोएपछि साबुन पानीले हात र औजार राम्ररी धुनुहोस्।",
                    "लाही किरा (Aphids) नियन्त्रण गर्नुहोस् र बारी नजिक सुर्तीजन्य पदार्थ प्रयोग नगर्नुहोस्।"
                ]
            },
            "Tomato_Yellow_Leaf_Curl_Virus": {
                "action": "सेतो झिंगा (Whiteflies) तुरुन्तै नियन्त्रण गर्नुहोस्।",
                "steps": [
                    "सेतो झिंगा नियन्त्रण गर्न नीमको तेल (Neem oil) वा पहेँलो पासो प्रयोग गर्नुहोस्।",
                    "साना बिरुवाहरूलाई मसिनो जाली (Insect net) ले ढाकेर राख्नुहोस्।",
                    "बारी वरपरका झारपातहरू सफा राख्नुहोस्।"
                ]
            },
            "healthy": {
                "action": "बिरुवा स्वस्थ छ — हालको हेरचाहलाई निरन्तरता दिनुहोस्!",
                "steps": [
                    "दैनिक ६–८ घण्टा प्रत्यक्ष घाम लाग्ने व्यवस्था गर्नुहोस्।",
                    "हप्तामा १–१.५ इन्च नियमित रूपमा गहिरो सिँचाइ गर्नुहोस्।",
                    "हरेक ३–४ हप्तामा सन्तुलित जैविक मल प्रयोग गर्नुहोस्।"
                ]
            }
        }
    }
}

# 3. Custom CSS & Bright Curved Header Box Styling
st.markdown("""
    <style>
    /* Hide Default Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}

    /* Sliding Sidebar Animation */
    section[data-testid="stSidebar"] {
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .sidebar-slide {
        animation: slideInLeft 0.55s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes slideInLeft {
        0% { transform: translateX(-35px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    /* Bright Curved Rectangle Header Box */
    .bright-header-box {
        background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 55%, #f1f8e9 100%);
        border: 2px solid #81c784;
        border-radius: 26px;
        padding: 20px 28px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        box-shadow: 0 12px 32px rgba(76, 175, 80, 0.16), inset 0 0 24px rgba(255, 255, 255, 0.9);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .bright-header-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 38px rgba(76, 175, 80, 0.24), inset 0 0 24px rgba(255, 255, 255, 1);
    }

    /* Compact Circular Logo */
    .logo-circle {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: #ffffff;
        border: 3px solid #2e7d32;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.22);
    }
    .logo-circle img {
        width: 100%;
        height: 100%;
        object-fit: cover;
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
        font-weight: 500;
        color: #388e3c;
        margin-top: 4px;
    }

    /* Interactive How It Works Cards */
    .how-section-title {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
    }
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }
    .step-card:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: #66bb6a;
        background: linear-gradient(180deg, #ffffff 0%, #f1f8e9 100%);
        box-shadow: 0 12px 24px rgba(46, 125, 50, 0.14);
    }
    .step-title {
        font-weight: 700;
        font-size: 0.98rem;
        color: #1b5e20;
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.4;
    }

    /* Side-by-Side Tabs */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        border-radius: 12px 12px 0 0 !important;
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
        font-size: 1.3rem;
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
    }

    /* Treatment Card */
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

# 5. Sliding Sidebar with Language Settings & Talk-with-AI Slider
with st.sidebar:
    st.markdown('<div class="sidebar-slide">', unsafe_allow_html=True)
    
    # Brand Mini Icon
    if tomato_b64:
        st.markdown(f'''
            <div style="text-align: center; margin-bottom: 10px;">
                <img src="{tomato_b64}" width="68" style="border-radius: 50%; border: 2px solid #2e7d32; box-shadow: 0 4px 10px rgba(0,0,0,0.12);">
            </div>
        ''', unsafe_allow_html=True)

    # Language Selector (Default English)
    if "app_lang" not in st.session_state:
        st.session_state.app_lang = "English"

    selected_lang = st.radio(
        "🌐 Language / भाषा",
        options=["English", "नेपाली"],
        index=0 if st.session_state.app_lang == "English" else 1,
        horizontal=True
    )
    st.session_state.app_lang = selected_lang
    t = TEXT[selected_lang]

    st.markdown("---")

    # Talk with AI Slider Option
    st.markdown(f"### {t['ai_slider_header']}")
    ai_slider = st.select_slider(
        t["ai_slider_label"],
        options=[t["slider_off"], "Gemini AI", "ChatGPT"],
        value=t["slider_off"]
    )

    if ai_slider == "Gemini AI":
        st.info(f"{t['ai_prompt_text']} **Google Gemini**")
        st.link_button(f"{t['ai_btn_text']} (Gemini)", "https://gemini.google.com", use_container_width=True)
    elif ai_slider == "ChatGPT":
        st.info(f"{t['ai_prompt_text']} **ChatGPT**")
        st.link_button(f"{t['ai_btn_text']} (ChatGPT)", "https://chatgpt.com", use_container_width=True)

    st.markdown("---")

    # Supported Categories in Active Language
    st.markdown(f"### {t['supported_header']}")
    if class_indices:
        for idx, name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            raw_key = name.replace("Tomato___", "")
            translated_name = t["diseases"].get(raw_key, raw_key.replace("_", " "))
            st.markdown(f"- **{translated_name}**")

    st.markdown('</div>', unsafe_allow_html=True)

# 6. Bright Curved Header Box (Circular Small Farmer Logo + "Plant Doctor")
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

# 7. Interactive "How Does It Work?" Space
st.markdown(f'<div class="how-section-title">{t["how_title"]}</div>', unsafe_allow_html=True)
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
            st.image(img_display, use_column_width=True)
            
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

        # Actionable Treatment Guide Card in Selected Language
        guide_info = t["guides"].get(disease_key, t["guides"].get("healthy"))
        if guide_info:
            st.markdown(f"""
                <div class="treatment-card">
                    <div class="treatment-title">{t['treatment_header']}</div>
                    <p style="margin-bottom: 10px; font-weight: 600; color: #1b5e20;">{t['primary_action']} {guide_info['action']}</p>
                    <ul style="margin-bottom: 0; padding-left: 20px; color: #444;">
                        {''.join([f'<li style="margin-bottom: 4px;">{step}</li>' for step in guide_info['steps']])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Expandable Probability Distribution
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(t["prob_expander"]):
            for idx, prob in enumerate(predictions[0]):
                raw_k = class_indices[str(idx)].replace("Tomato___", "")
                cat_name = t["diseases"].get(raw_k, raw_k.replace("_", " "))
                st.progress(float(prob), text=f"{cat_name}: {prob*100:.2f}%")
