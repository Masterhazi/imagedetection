import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Use caching to load the model only once
@st.cache_resource
def load_yolo_model():
    # Use the lightweight nano model
    return YOLO("yolo11n.pt")

st.title("Hazi's YOLO AI Masterclass")
st.write("Take a picture to see real-time object detection!")

model = load_yolo_model()

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # 1. Convert buffer to PIL Image
    img = Image.open(img_file_buffer)
    
    # 2. Convert to Numpy (RGB)
    frame_rgb = np.array(img)
    
    # 3. YOLO expects BGR, so we swap R and B channels using Numpy
    frame_bgr = frame_rgb[:, :, ::-1] 
    
    # 4. Run YOLO
    results = model(frame_bgr)
    
    # 5. Get annotated frame (Numpy array in BGR)
    annotated_frame_bgr = results[0].plot()
    
    # 6. Swap back to RGB for Streamlit display
    annotated_frame_rgb = annotated_frame_bgr[:, :, ::-1]
    
    # 7. Show it
    st.image(annotated_frame_rgb, caption="Detections")

    # Raw data output
    for box in results[0].boxes:
        st.write(f"Detected: **{model.names[int(box.cls[0])]}** ({float(box.conf[0]):.2f})")
