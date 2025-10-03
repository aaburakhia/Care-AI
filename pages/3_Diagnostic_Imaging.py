import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import os

# --- Page Configuration ---
st.set_page_config(page_title="Diagnostic Imaging Assistant", page_icon="🩻")
st.title("AI-Assisted Diagnostic Imaging")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()

# --- Model Loading ---
@st.cache_resource
def load_onnx_model():
    repo_id = "aaburakhia/Pneumonia-Detector-CareAI" 
    model_filename = "model.onnx"
    local_dir = "model"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, model_filename)
    
    if not os.path.exists(model_path):
        try:
            hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=local_dir)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()
    
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        st.stop()

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image):
    img_resized = image.resize((150, 150)).convert('RGB')
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Main App ---
session = load_onnx_model()

if session:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    st.write("Upload a chest X-ray image to get a preliminary analysis for the presence of pneumonia.")
    st.info("This model is a proof-of-concept and should **not** be used for actual medical diagnosis.", icon="⚠️")
    
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpeg", "jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Model is analyzing the image..."):
                processed_image = preprocess_image(image)
                
                outputs = session.run([output_name], {input_name: processed_image})
                prediction = outputs[0]
                
                confidence_score = float(prediction[0][0]) * 100
                
                st.success("Analysis Complete!")
                
                if confidence_score > 50:
                    st.warning(f"Model's Preliminary Finding: **Pneumonia Likely Detected**")
                else:
                    st.success(f"Model's Preliminary Finding: **Pneumonia Not Detected**")
                
                st.write(f"Confidence Score: **{confidence_score:.2f}%**")
