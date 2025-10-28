import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import os
from style_utils import add_custom_css

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Care-AI | Diagnostic Imaging",
    page_icon="🩻",
    layout="wide"
)
add_custom_css()

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()

# --- Scalable Model Loading ---
@st.cache_resource
def load_model(model_filename):
    """Generic function to download and load any ONNX model from Hugging Face."""
    repo_id = "aaburakhia/Pneumonia-Detector-CareAI"
    local_dir = "models"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, model_filename)
    
    if not os.path.exists(model_path):
        try:
            with st.spinner(f"Downloading {model_filename}... (This happens only once)"):
                hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=local_dir)
        except Exception as e:
            st.error(f"Error downloading model '{model_filename}': {e}")
            return None
    
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"Failed to load ONNX model '{model_filename}': {e}")
        return None

# --- Scalable Image Preprocessing ---
def preprocess_image(image: Image.Image, input_size=(150, 150)):
    """Generic function to preprocess an image for a model."""
    img_resized = image.resize(input_size).convert('RGB')
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Sidebar ---
with st.sidebar:
    st.header("About This Hub")
    st.markdown("""
    This Care-AI tool uses several deep learning models to analyze different types of medical images. Select a demo or upload your own image, then choose the appropriate analysis to perform.
    """)
    with st.expander("Model Details"):
        st.markdown("""
        **Pneumonia Model:**
        - **Architecture:** VGG16 Transfer Learning
        - **Input:** 150x150 RGB
        
        **Breast Cancer Model:**
        - **Architecture:** (e.g., ResNet, VGG - *update as needed*)
        - **Input:** 150x150 RGB
        """)
    st.info("This is a proof-of-concept and not a substitute for professional medical diagnosis.")


# --- Main Page Layout ---
st.title("Care-AI Diagnostic Imaging Hub")
st.divider()

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Provide an Image")
        uploaded_file = st.file_uploader("Upload a medical image", type=["jpeg", "jpg", "png"], label_visibility="collapsed")
        
        st.markdown("<h5 style='text-align: center; color: grey;'>or select a demo</h5>", unsafe_allow_html=True)
        
        # --- New Demo Buttons ---
        st.subheader("Pneumonia Demos (X-Ray)")
        p_col1, p_col2 = st.columns(2)
        if p_col1.button("Load Normal X-Ray", use_container_width=True):
            st.session_state.current_image_path = "demo_images/normal_1.jpeg"
            st.session_state.current_image_type = "Demo - Normal X-Ray"

        if p_col2.button("Load Pneumonia X-Ray", use_container_width=True):
            st.session_state.current_image_path = "demo_images/pneumonia_1.jpeg"
            st.session_state.current_image_type = "Demo - Pneumonia X-Ray"

        st.subheader("Breast Cancer Demos (Mammogram)")
        bc_col1, bc_col2 = st.columns(2)
        if bc_col1.button("Load Benign Scan", use_container_width=True):
            st.session_state.current_image_path = "demo_images/bc_benign_1.jpeg"
            st.session_state.current_image_type = "Demo - Benign Scan"

        if bc_col2.button("Load Malignant Scan", use_container_width=True):
            st.session_state.current_image_path = "demo_images/bc_malignant_1.jpeg"
            st.session_state.current_image_type = "Demo - Malignant Scan"

# --- Analysis & Results Column ---
with col2:
    with st.container(border=True):
        st.markdown("#### 2. Analyze & Review")
        
        image_to_analyze = None
        if uploaded_file:
            image_to_analyze = Image.open(uploaded_file)
            st.session_state.current_image_type = "Uploaded Image"
        elif 'current_image_path' in st.session_state:
            try:
                image_to_analyze = Image.open(st.session_state.current_image_path)
            except FileNotFoundError:
                st.error(f"Demo image not found at {st.session_state.current_image_path}. Please check the file exists in the 'demo_images' folder.")
                image_to_analyze = None
        
        if image_to_analyze:
            st.image(image_to_analyze, caption=st.session_state.current_image_type, use_container_width=True)
            st.markdown("---")
            st.write("Choose which analysis to perform:")

            # --- Analysis Buttons ---
            analysis_col1, analysis_col2 = st.columns(2)

            if analysis_col1.button("🔬 Analyze for Pneumonia", type="primary", use_container_width=True):
                with st.spinner("Care-AI is analyzing for Pneumonia..."):
                    session = load_model("model.onnx")
                    processed_image = preprocess_image(image_to_analyze)
                    
                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name
                    outputs = session.run([output_name], {input_name: processed_image})
                    score = float(outputs[0][0][0])
                    
                    st.subheader("Pneumonia Analysis Results")
                    if score > 0.5:
                        st.error(f"**Finding:** Pneumonia Likely Detected (Confidence: {score*100:.2f}%)", icon="🏥")
                    else:
                        st.success(f"**Finding:** Pneumonia Not Detected (Confidence: {(1-score)*100:.2f}%)", icon="🛡️")

            if analysis_col2.button("🎀 Analyze for Breast Cancer", use_container_width=True):
                with st.spinner("Care-AI is analyzing for Breast Cancer..."):
                    session = load_model("breast_cancer_classifier.onnx")
                    processed_image = preprocess_image(image_to_analyze) # Assumes 150x150 input

                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name
                    outputs = session.run([output_name], {input_name: processed_image})
                    score = float(outputs[0][0][0])
                    
                    st.subheader("Breast Cancer Analysis Results")
                    # IMPORTANT: This assumes score > 0.5 means MALIGNANT. Adjust if needed.
                    if score > 0.5:
                        st.error(f"**Finding:** Malignant Cells Likely Detected (Confidence: {score*100:.2f}%)", icon="⚠️")
                    else:
                        st.success(f"**Finding:** Benign Cells Detected (Confidence: {(1-score)*100:.2f}%)", icon="✅")

        else:
            st.info("Upload an image or select a demo to begin.")
