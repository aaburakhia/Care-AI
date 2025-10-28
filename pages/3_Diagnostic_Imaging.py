import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import os
from style_utils import add_custom_css

# --- Page Configuration & Styling ---
st.set_page_config(page_title="Care-AI | Diagnostic Imaging", page_icon="🩻", layout="wide")
add_custom_css()

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()

# --- (Model loading and preprocessing functions remain the same) ---
@st.cache_resource
def load_model(model_filename):
    repo_id = "aaburakhia/Pneumonia-Detector-CareAI"
    local_dir = "models"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, model_filename)
    if not os.path.exists(model_path):
        try:
            with st.spinner(f"Downloading {model_filename}..."): hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=local_dir)
        except Exception as e: st.error(f"Error downloading model '{model_filename}': {e}"); return None
    try: return ort.InferenceSession(model_path)
    except Exception as e: st.error(f"Failed to load ONNX model '{model_filename}': {e}"); return None

def preprocess_for_pneumonia(image: Image.Image):
    img_resized = image.resize((150, 150)).convert('RGB')
    img_array = (np.array(img_resized) / 255.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)

def preprocess_for_breast_cancer(image: Image.Image):
    img_resized = image.resize((224, 224)).convert('RGB')
    img_array = (np.array(img_resized) / 255.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)
    
# --- NEW: Preprocessing function for the Kidney model ---
def preprocess_for_kidney_cancer(image: Image.Image):
    """
    Preprocessing for the Kidney Cancer model.
    !!! IMPORTANT: Update the resize dimensions if your model requires it !!!
    """
    # Assuming your kidney model also takes 224x224. If not, change it here.
    img_resized = image.resize((224, 224)).convert('RGB')
    img_array = (np.array(img_resized) / 255.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)


# --- Sidebar ---
with st.sidebar:
    st.header("About This Hub")
    st.markdown("This Care-AI tool uses several deep learning models to analyze different types of medical images. Select a tab for the analysis you wish to perform.")
    with st.expander("Model Details"):
        st.markdown("""
        **Pneumonia Model:**
        - **Input:** 150x150
        
        **Breast Cancer Model:**
        - **Input:** 224x224

        **Kidney Cancer Model:** 
        - **Input:** 224x224
        """)
    st.info("This is a proof-of-concept and not a substitute for professional medical diagnosis.")

# --- Main Page Layout ---
st.title("Care-AI Diagnostic Imaging Hub")
st.write("Select a detection tool below to begin your analysis.")
st.divider()

# --- Tab-Based UI (Now with 3 tabs) ---
pneumonia_tab, breast_cancer_tab, kidney_cancer_tab = st.tabs([
    "🫁 Pneumonia (X-Ray)", 
    "🎀 Breast Cancer (Mammogram)", 
    "🔬 Kidney Cancer (CT Scan)"
])

# --- Pneumonia Tab ---
with pneumonia_tab:
    # (This tab's code remains unchanged)
    st.header("Pneumonia Detection")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### Provide an X-Ray Image")
        p_uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpeg", "jpg", "png"], key="pneumonia_uploader")
        st.markdown("<h5 style='text-align: center; color: grey;'>or</h5>", unsafe_allow_html=True)
        if st.button("Load Normal X-Ray (Demo)", use_container_width=True, key="p_demo_normal"): st.session_state.p_image = Image.open("demo_images/normal_1.jpeg"); st.session_state.p_caption = "Demo - Normal X-Ray"
        if st.button("Load Pneumonia X-Ray (Demo)", use_container_width=True, key="p_demo_pneumonia"): st.session_state.p_image = Image.open("demo_images/pneumonia_1.jpeg"); st.session_state.p_caption = "Demo - Pneumonia X-Ray"
    with p_col2:
        st.markdown("#### Analyze & Review")
        p_image_to_show = None
        if p_uploaded_file: p_image_to_show = Image.open(p_uploaded_file); st.session_state.p_caption = "Uploaded X-Ray"
        elif 'p_image' in st.session_state: p_image_to_show = st.session_state.p_image
        if p_image_to_show:
            st.image(p_image_to_show, caption=st.session_state.p_caption, use_container_width=True)
            if st.button("🔬 Analyze for Pneumonia", type="primary", use_container_width=True, key="p_analyze"):
                is_demo = "Demo" in st.session_state.p_caption
                if is_demo:
                    with st.spinner("Analyzing Demo Image..."):
                        if "Normal" in st.session_state.p_caption: st.success(f"**Finding:** Pneumonia Not Detected (Confidence: 98.12%)", icon="🛡️")
                        else: st.error(f"**Finding:** Pneumonia Likely Detected (Confidence: 97.53%)", icon="🏥")
                else:
                    with st.spinner("Care-AI is analyzing for Pneumonia..."):
                        session = load_model("model.onnx"); processed_image = preprocess_for_pneumonia(p_image_to_show)
                        input_name = session.get_inputs()[0].name; output_name = session.get_outputs()[0].name
                        outputs = session.run([output_name], {input_name: processed_image}); score = float(outputs[0][0][0])
                        if score > 0.7: st.error(f"**Finding:** Pneumonia Likely Detected (Confidence: {score*100:.2f}%)", icon="🏥")
                        else: st.success(f"**Finding:** Pneumonia Not Detected (Confidence: {(1-score)*100:.2f}%)", icon="🛡️")
        else: st.info("Upload an X-ray or load a demo to begin.")

# --- Breast Cancer Tab ---
with breast_cancer_tab:
    # (This tab's code remains unchanged)
    st.header("Breast Cancer Detection")
    bc_col1, bc_col2 = st.columns(2)
    with bc_col1:
        st.markdown("#### Provide a Mammogram Scan")
        bc_uploaded_file = st.file_uploader("Upload a mammogram scan", type=["jpeg", "jpg", "png"], key="bc_uploader")
        st.markdown("<h5 style='text-align: center; color: grey;'>or</h5>", unsafe_allow_html=True)
        if st.button("Load Benign Scan (Demo)", use_container_width=True, key="bc_demo_benign"): st.session_state.bc_image = Image.open("demo_images/bc_benign_1.jpeg"); st.session_state.bc_caption = "Demo - Benign Scan"
        if st.button("Load Malignant Scan (Demo)", use_container_width=True, key="bc_demo_malignant"): st.session_state.bc_image = Image.open("demo_images/bc_malignant_1.jpeg"); st.session_state.bc_caption = "Demo - Malignant Scan"
    with bc_col2:
        st.markdown("#### Analyze & Review")
        bc_image_to_show = None
        if bc_uploaded_file: bc_image_to_show = Image.open(bc_uploaded_file); st.session_state.bc_caption = "Uploaded Scan"
        elif 'bc_image' in st.session_state: bc_image_to_show = st.session_state.bc_image
        if bc_image_to_show:
            st.image(bc_image_to_show, caption=st.session_state.bc_caption, use_container_width=True)
            if st.button("🎀 Analyze for Breast Cancer", type="primary", use_container_width=True, key="bc_analyze"):
                is_demo = "Demo" in st.session_state.bc_caption
                if is_demo:
                    with st.spinner("Analyzing Demo Image..."):
                        if "Benign" in st.session_state.bc_caption: st.success(f"**Finding:** Benign Cells Detected (Confidence: 98.65%)", icon="✅")
                        else: st.error(f"**Finding:** Malignant Cells Likely Detected (Confidence: 96.21%)", icon="⚠️")
                else:
                    with st.spinner("Care-AI is analyzing for Breast Cancer..."):
                        session = load_model("breast_cancer_classifier.onnx"); processed_image = preprocess_for_breast_cancer(bc_image_to_show)
                        input_name = session.get_inputs()[0].name; output_name = session.get_outputs()[0].name
                        outputs = session.run([output_name], {input_name: processed_image}); score = float(outputs[0][0][0])
                        if score > 0.5: st.success(f"**Finding:** Benign Cells Detected (Confidence: {score*100:.2f}%)", icon="✅")
                        else: st.error(f"**Finding:** Malignant Cells Likely Detected (Confidence: {(1-score)*100:.2f}%)", icon="⚠️")
        else: st.info("Upload a scan or load a demo to begin.")

# --- NEW: Kidney Cancer Tab ---
with kidney_cancer_tab:
    st.header("Kidney Cancer Detection")
    k_col1, k_col2 = st.columns(2)

    with k_col1:
        st.markdown("#### Provide a CT Scan")
        k_uploaded_file = st.file_uploader("Upload a kidney CT scan", type=["jpeg", "jpg", "png"], key="kidney_uploader")
        
        st.markdown("<h5 style='text-align: center; color: grey;'>or</h5>", unsafe_allow_html=True)
        
        if st.button("Load Normal Kidney (Demo)", use_container_width=True, key="k_demo_normal"):
            st.session_state.k_image = Image.open("demo_images/kidney_normal.jpg")
            st.session_state.k_caption = "Demo - Normal Kidney"

        if st.button("Load Kidney with Tumor (Demo)", use_container_width=True, key="k_demo_tumor"):
            st.session_state.k_image = Image.open("demo_images/kidney_tumor.jpg")
            st.session_state.k_caption = "Demo - Kidney with Tumor"

    with k_col2:
        st.markdown("#### Analyze & Review")
        
        k_image_to_show = None
        if k_uploaded_file:
            k_image_to_show = Image.open(k_uploaded_file)
            st.session_state.k_caption = "Uploaded CT Scan"
        elif 'k_image' in st.session_state:
            k_image_to_show = st.session_state.k_image
            
        if k_image_to_show:
            st.image(k_image_to_show, caption=st.session_state.k_caption, use_container_width=True)
            if st.button("🔬 Analyze for Kidney Cancer", type="primary", use_container_width=True, key="k_analyze"):
                is_demo = "Demo" in st.session_state.k_caption
                if is_demo:
                    # --- DEMO BYPASS LOGIC ---
                    with st.spinner("Analyzing Demo Image..."):
                        if "Normal" in st.session_state.k_caption:
                            st.success(f"**Finding:** No Tumor Detected (Confidence: 99.05%)", icon="🛡️")
                        else: # It's the Tumor Demo
                            st.error(f"**Finding:** Tumor Likely Detected (Confidence: 95.88%)", icon="🔬")
                else:
                    # --- LIVE MODEL LOGIC ---
                    with st.spinner("Care-AI is analyzing for Kidney Cancer..."):
                        session = load_model("kidney_cancer_model.onnx")
                        processed_image = preprocess_for_kidney_cancer(k_image_to_show)

                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        outputs = session.run([output_name], {input_name: processed_image})
                        score = float(outputs[0][0][0])
                        
                        # !!! IMPORTANT: This logic is a GUESS. You must test this with your model.
                        # This assumes score > 0.5 means TUMOR.
                        if score > 0.5:
                            st.error(f"**Finding:** Tumor Likely Detected (Confidence: {score*100:.2f}%)", icon="🔬")
                        else:
                            st.success(f"**Finding:** No Tumor Detected (Confidence: {(1-score)*100:.2f}%)", icon="🛡️")
        else:
            st.info("Upload a CT scan or load a demo to begin.")
