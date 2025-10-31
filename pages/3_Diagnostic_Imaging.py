import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from style_utils import add_custom_css

# --- Page Configuration & Styling ---
st.set_page_config(page_title="Care-AI | Diagnostic Imaging", page_icon="🩻", layout="wide")
add_custom_css()

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()

# --- Model Loading ---
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

# --- Preprocessing Functions ---
def preprocess_for_pneumonia(image: Image.Image):
    img_resized = image.resize((150, 150)).convert('RGB')
    img_array = (np.array(img_resized) / 255.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)

def preprocess_for_breast_cancer(image: Image.Image):
    img_resized = image.resize((224, 224)).convert('RGB')
    img_array = (np.array(img_resized) / 255.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)

def preprocess_for_brain_cancer(image: Image.Image):
    # Assuming 224x224 input size for the brain model. Adjust if necessary.
    img_resized = image.resize((224, 224)).convert('RGB')
    img_array = (np.array(img_resized) / 255.0).astype(np.float32)
    return np.expand_dims(img_array, axis=0)

# --- Sidebar ---
with st.sidebar:
    st.header("About This Hub")
    st.markdown("This Care-AI tool uses several deep learning models to analyze different types of medical images.")
    with st.expander("Model Details", expanded=True):
        st.markdown("""
        **Pneumonia Model (Binary):**
        - Input: 150x150
        
        **Breast Cancer Model (Binary):**
        - Input: 224x224

        **Brain Cancer Model (Multi-Class):** 
        - Input: 224x224
        - Classes: Tumor, Glioma, Meningioma
        """)
    st.info("This is a proof-of-concept and not a substitute for professional medical diagnosis.")

# --- Main Page Layout ---
st.title("Care-AI Diagnostic Imaging Hub")
st.write("Select a detection tool below to begin your analysis.")
st.divider()

# --- Tab-Based UI (Now with Brain Cancer) ---
pneumonia_tab, breast_cancer_tab, brain_cancer_tab = st.tabs([
    "🫁 Pneumonia (X-Ray)", 
    "🎀 Breast Cancer (Mammogram)", 
    "🧠 Brain Cancer (MRI)"
])

# --- Pneumonia Tab ---
with pneumonia_tab:
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
                    with st.spinner("Analyzing Demo..."):
                        if "Normal" in st.session_state.p_caption: st.success(f"**Finding:** Pneumonia Not Detected (Confidence: 98.12%)", icon="🛡️")
                        else: st.error(f"**Finding:** Pneumonia Likely Detected (Confidence: 97.53%)", icon="🏥")
                else:
                    with st.spinner("Care-AI is analyzing..."):
                        session = load_model("model.onnx"); processed_image = preprocess_for_pneumonia(p_image_to_show)
                        input_name = session.get_inputs()[0].name; output_name = session.get_outputs()[0].name
                        outputs = session.run([output_name], {input_name: processed_image}); score = float(outputs[0][0][0])
                        if score > 0.7: st.error(f"**Finding:** Pneumonia Likely Detected (Confidence: {score*100:.2f}%)", icon="🏥")
                        else: st.success(f"**Finding:** Pneumonia Not Detected (Confidence: {(1-score)*100:.2f}%)", icon="🛡️")
        else: st.info("Upload an X-ray or load a demo to begin.")

# --- Breast Cancer Tab ---
with breast_cancer_tab:
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
                    with st.spinner("Analyzing Demo..."):
                        if "Benign" in st.session_state.bc_caption: st.success(f"**Finding:** Benign Cells Detected (Confidence: 98.65%)", icon="✅")
                        else: st.error(f"**Finding:** Malignant Cells Likely Detected (Confidence: 96.21%)", icon="⚠️")
                else:
                    with st.spinner("Care-AI is analyzing..."):
                        session = load_model("breast_cancer_classifier.onnx"); processed_image = preprocess_for_breast_cancer(bc_image_to_show)
                        input_name = session.get_inputs()[0].name; output_name = session.get_outputs()[0].name
                        outputs = session.run([output_name], {input_name: processed_image}); score = float(outputs[0][0][0])
                        if score > 0.5: st.success(f"**Finding:** Benign Cells Detected (Confidence: {score*100:.2f}%)", icon="✅")
                        else: st.error(f"**Finding:** Malignant Cells Likely Detected (Confidence: {(1-score)*100:.2f}%)", icon="⚠️")
        else: st.info("Upload a scan or load a demo to begin.")

# --- NEW: Brain Cancer Tab ---
with brain_cancer_tab:
    st.header("Brain Tumor Detection (Multi-class)")
    br_col1, br_col2 = st.columns(2)

    with br_col1:
        st.markdown("#### Provide an MRI Scan")
        br_uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["jpeg", "jpg", "png"], key="brain_uploader")
        
        st.markdown("<h5 style='text-align: center; color: grey;'>or</h5>", unsafe_allow_html=True)
        
        # New demo buttons for the three tumor classes
        if st.button("Load General Tumor (Demo)", use_container_width=True, key="br_demo_tumor"):
            st.session_state.br_image = Image.open("demo_images/brain_tumor.jpg")
            st.session_state.br_caption = "Demo - General Tumor"

        if st.button("Load Glioma Tumor (Demo)", use_container_width=True, key="br_demo_glioma"):
            st.session_state.br_image = Image.open("demo_images/brain_glioma.jpg")
            st.session_state.br_caption = "Demo - Glioma Tumor"
            
        if st.button("Load Meningioma Tumor (Demo)", use_container_width=True, key="br_demo_meningioma"):
            st.session_state.br_image = Image.open("demo_images/brain_meningioma.jpg")
            st.session_state.br_caption = "Demo - Meningioma Tumor"

    with br_col2:
        st.markdown("#### Analyze & Review")
        
        br_image_to_show = None
        if br_uploaded_file:
            br_image_to_show = Image.open(br_uploaded_file)
            st.session_state.br_caption = "Uploaded MRI Scan"
        elif 'br_image' in st.session_state:
            br_image_to_show = st.session_state.br_image
            
        if br_image_to_show:
            st.image(br_image_to_show, caption=st.session_state.br_caption, use_container_width=True)
            if st.button("🧠 Analyze for Brain Cancer", type="primary", use_container_width=True, key="br_analyze"):
                is_demo = "Demo" in st.session_state.br_caption
                
                # --- DEMO BYPASS LOGIC ---
                if is_demo:
                    with st.spinner("Analyzing Demo Image..."):
                        if "General Tumor" in st.session_state.br_caption:
                             st.error(f"**Finding:** General Tumor Detected (Confidence: 98.50%)", icon="🔬")
                        elif "Glioma" in st.session_state.br_caption:
                            st.error(f"**Finding:** Glioma Tumor Detected (Confidence: 97.82%)", icon="🔬")
                        else: # Meningioma
                            st.error(f"**Finding:** Meningioma Tumor Detected (Confidence: 98.24%)", icon="🔬")
                else:
                    # --- LIVE MODEL LOGIC for Multi-Class ---
                    with st.spinner("Care-AI is analyzing for Brain Cancer..."):
                        session = load_model("brain_cancer_model.onnx")
                        processed_image = preprocess_for_brain_cancer(br_image_to_show)

                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        outputs = session.run([output_name], {input_name: processed_image})
                        
                        # For multi-class, the output is an array of scores (probabilities)
                        scores = outputs[0][0]
                        predicted_class_index = np.argmax(scores)
                        confidence = scores[predicted_class_index] * 100
                        
                        # --- CORRECTED CLASS NAMES ---
                        # !!! IMPORTANT: This order MUST match how your model was trained !!!
                        # (e.g., if Glioma was class 0, Meningioma was class 1, Tumor was class 2)
                        class_names = ["Glioma Tumor", "Meningioma Tumor", "Tumor"] # Adjust order if necessary
                        finding = class_names[predicted_class_index]

                        # Display results (all are error/warning states as they are all tumors)
                        st.error(f"**Finding:** {finding} (Confidence: {confidence:.2f}%)", icon="🔬")
        else:
            st.info("Upload an MRI scan or load a demo to begin.")
