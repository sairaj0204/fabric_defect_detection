import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Fabric Defect Detection", page_icon="🏭", layout="wide")

# --- 2. LOAD MODEL (Cached for performance) ---
@st.cache_resource
def load_model():
    # Ensure best.pt is in the same directory as this script!
    # If it's still in the runs folder, use: "runs/detect/fabric_defect_run/weights/best.pt"
    model_path = "best.pt" 
    if not os.path.exists(model_path):
        # Fallback to your original path if you haven't moved it
        model_path = r"runs\detect\fabric_defect_run\weights\best.pt"
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def process_image(image, conf=0.4):
    """Runs YOLO inference and returns the plotted RGB image."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # YOLO expects BGR or RGB depending on the source, but handles PIL/numpy well.
    results = model.predict(source=image, conf=conf, verbose=False)
    # Plot returns a BGR numpy array
    annotated_bgr = results[0].plot()
    # Convert BGR to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb

# --- 4. UI HEADER ---
st.title("🏭 AI Fabric Defect Detection System")
st.markdown("**Real-time manufacturing quality control powered by YOLOv8.**")

# --- 5. SIDEBAR NAVIGATION ---
app_mode = st.sidebar.radio("Choose Mode", ["Single Image", "Webcam Capture", "Bulk Processing"])

# ==========================================
# MODE 1: SINGLE IMAGE
# ==========================================
if app_mode == "Single Image":
    st.header("Single Image Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        # Option to upload or use a sample image
        source_type = st.radio("Select Input Source:", ["Upload Image", "Use Sample Image"])
        
        input_image = None
        if source_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
        else:
            # Load sample images from directory
            sample_dir = "sample_images"
            if os.path.exists(sample_dir) and os.listdir(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                selected_sample = st.selectbox("Choose a sample:", sample_files)
                if selected_sample:
                    input_image = Image.open(os.path.join(sample_dir, selected_sample))
            else:
                st.warning("⚠️ Please create a 'sample_images' folder with images to use this feature.")

        if input_image:
            st.image(input_image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        if input_image:
            with st.spinner("Analyzing..."):
                result_img = process_image(input_image)
                st.image(result_img, caption="Detected Defects", use_container_width=True)
        else:
            st.info("Upload an image or select a sample to see the results here.")

# ==========================================
# MODE 2: WEBCAM CAPTURE
# ==========================================
elif app_mode == "Webcam Capture":
    st.header("Live Webcam Analysis")
    st.markdown("*Note: Your browser may prompt you to allow camera access.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Camera Input")
        # st.camera_input is highly reliable and uses browser APIs
        cam_image = st.camera_input("Take a picture")
        
        if cam_image:
            # Convert the uploaded camera file to a PIL Image
            img = Image.open(cam_image)
            st.image(img, caption="Captured Image", use_container_width=True)
            
    with col2:
        st.subheader("Detection Result")
        if cam_image:
            with st.spinner("Analyzing..."):
                result_img = process_image(img)
                st.image(result_img, caption="Detected Defects", use_container_width=True)
        else:
            st.info("Allow camera access and take a picture to see results.")

# ==========================================
# MODE 3: BULK PROCESSING
# ==========================================
elif app_mode == "Bulk Processing":
    st.header("Bulk Batch Processing")
    st.markdown("Upload multiple fabric images at once to simulate high-throughput conveyor belt scanning.")
    
    uploaded_files = st.file_uploader("Upload Image Batch", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Analyze Batch", type="primary"):
            st.write(f"Processing {len(uploaded_files)} images...")
            
            # Create a grid layout for results
            cols = st.columns(3) 
            
            for idx, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                
                # Run inference
                result_img = process_image(img)
                
                # Display in grid (3 columns)
                with cols[idx % 3]:
                    st.image(result_img, caption=f"Result: {uploaded_file.name}", use_container_width=True)