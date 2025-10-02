import streamlit as st
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
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
def load_tflite_model():
    """
    Downloads the TFLite model from Hugging Face Hub and loads the interpreter.
    """

    repo_id = "aaburakhia/Pneumonia-Detector-CareAI" 
    model_filename = "model.tflite"

    # To make the download path more robust, we'll store it in a dedicated folder
    local_dir = "model"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, model_filename)

    # Download only if the model doesn't exist locally
    if not os.path.exists(model_path):
        try:
            hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=local_dir)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load TFLite model: {e}")
        st.stop()

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image, input_details):
    """
    Prepares the uploaded image to be compatible with the TFLite model.
    """
    _, height, width, _ = input_details[0]['shape']
    
    img_resized = image.resize((width, height))
    img_rgb = img_resized.convert('RGB')
    
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)
    
    return img_expanded

# --- Main App ---
interpreter = load_tflite_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    st.write("Upload a chest X-ray image to get a preliminary analysis for the presence of pneumonia.")
    st.info("This model is a proof-of-concept and should **not** be used for actual medical diagnosis.", icon="⚠️")

    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Model is analyzing the image..."):
                processed_image = preprocess_image(image, input_details)
                
                interpreter.set_tensor(input_details[0]['index'], processed_image)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                confidence_score = float(prediction[0][0]) * 100
                
                st.success("Analysis Complete!")
                
                if confidence_score > 50:
                    st.warning(f"Model's Preliminary Finding: **Pneumonia Likely Detected**")
                else:
                    st.success(f"Model's Preliminary Finding: **Pneumonia Not Detected**")
                
                st.write(f"Confidence Score: **{confidence_score:.2f}%**")
