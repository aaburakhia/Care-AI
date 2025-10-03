import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import os

# Page Configuration
st.set_page_config(
    page_title="Diagnostic Imaging Assistant",
    page_icon="🩻",
    layout="wide"
)

# Authentication Check
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()

# Model Loading
@st.cache_resource
def load_onnx_model():
    """Load the ONNX model from Hugging Face"""
    repo_id = "aaburakhia/Pneumonia-Detector-CareAI"
    model_filename = "model.onnx"
    local_dir = "model"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, model_filename)
    
    if not os.path.exists(model_path):
        try:
            with st.spinner("Downloading model from Hugging Face..."):
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

# Image Preprocessing
def preprocess_image(image: Image.Image):
    """Preprocess image for model input"""
    img_resized = image.resize((150, 150)).convert('RGB')
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Main App
st.title("🩻 AI-Assisted Diagnostic Imaging")
st.markdown("### Pneumonia Detection from Chest X-Rays")

# Load model
session = load_onnx_model()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Information banner
st.info(
    "⚠️ **Important Notice**: This model is a proof-of-concept and should NOT be used for actual medical diagnosis. "
    "Always consult with qualified healthcare professionals.",
    icon="⚠️"
)

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=["jpeg", "jpg", "png"],
        help="Upload a clear chest X-ray image"
    )
    
    # Demo images section
    st.markdown("---")
    st.markdown("#### Try Demo Images")
    
    # Check if demo images exist
    demo_available = os.path.exists("demo_images") and len(os.listdir("demo_images")) > 0
    
    if demo_available:
        st.write("Don't have an X-ray? Try one of our sample images:")
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            if st.button("Load Normal X-Ray", use_container_width=True):
                try:
                    st.session_state.demo_image = Image.open("demo_images/normal_1.jpeg")
                    st.session_state.demo_type = "Normal"
                    st.success("Normal X-ray loaded!")
                except:
                    st.error("Could not load demo image")
        
        with demo_col2:
            if st.button("Load Pneumonia X-Ray", use_container_width=True):
                try:
                    st.session_state.demo_image = Image.open("demo_images/pneumonia_1.jpeg")
                    st.session_state.demo_type = "Pneumonia"
                    st.success("Pneumonia X-ray loaded!")
                except:
                    st.error("Could not load demo image")
    else:
        st.info("Demo images not available. Please upload your own X-ray image.")

with col2:
    st.markdown("#### Analysis Results")
    
    # Determine which image to use
    image_to_analyze = None
    image_source = None
    
    if uploaded_file is not None:
        image_to_analyze = Image.open(uploaded_file)
        image_source = "Uploaded X-Ray"
    elif 'demo_image' in st.session_state:
        image_to_analyze = st.session_state.demo_image
        image_source = f"Demo X-Ray ({st.session_state.demo_type})"
    
    if image_to_analyze is not None:
        st.image(image_to_analyze, caption=image_source, use_container_width=True)
        
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing X-ray image..."):
                # Preprocess and predict
                processed_image = preprocess_image(image_to_analyze)
                outputs = session.run([output_name], {input_name: processed_image})
                prediction = outputs[0]
                
                confidence_score = float(prediction[0][0]) * 100
                
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Display result
                if confidence_score > 50:
                    st.error(
                        f"**Preliminary Finding: Pneumonia Likely Detected**\n\n"
                        f"Confidence: **{confidence_score:.2f}%**",
                        icon="⚠️"
                    )
                    st.markdown("""
                    **Recommended Actions:**
                    - Consult with a radiologist immediately
                    - Schedule follow-up imaging if recommended
                    - Discuss treatment options with your healthcare provider
                    """)
                else:
                    st.success(
                        f"**Preliminary Finding: Pneumonia Not Detected**\n\n"
                        f"Confidence: **{100 - confidence_score:.2f}%**",
                        icon="✅"
                    )
                    st.markdown("""
                    **Note:**
                    - This suggests no obvious signs of pneumonia
                    - Regular check-ups are still recommended
                    - Consult a doctor if symptoms persist
                    """)
                
                # Confidence meter
                st.markdown("#### Confidence Meter")
                st.progress(confidence_score / 100)
                
                st.markdown("---")
                st.caption(
                    "This analysis is for educational purposes only. Medical decisions should "
                    "always be made in consultation with qualified healthcare professionals."
                )
    else:
        st.info("👆 Upload an X-ray image or load a demo image to begin analysis")

# Sidebar
with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown("""
    This diagnostic imaging assistant uses a deep learning model trained on chest X-ray images.
    
    **Model Details:**
    - Architecture: VGG16 Transfer Learning
    - Input: 150x150 RGB chest X-ray
    - Output: Binary classification
    - Format: ONNX (optimized)
    
    **Performance:**
    - Training Accuracy: ~95%
    - Validation Accuracy: ~93%
    - Test Accuracy: ~90%
    
    **Limitations:**
    - Not a replacement for professional diagnosis
    - May produce false positives/negatives
    - Performance varies with image quality
    """)
    
    st.markdown("---")
    st.markdown("### Privacy Notice")
    st.markdown("""
    - Images processed locally
    - No data stored permanently
    - Results not saved
    """)
