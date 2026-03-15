import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Use caching to load the model only once
@st.cache_resource
def load_yolo_model():
    model = YOLO("yolov8n.pt")
    return model

# --- App UI ---
st.title("Hazi's YOLO AI Masterclass")
st.write("Take a picture to see real-time object detection!")

# Load the model from the cached function
model = load_yolo_model()

# Use Streamlit's built-in webcam input
img_file_buffer = st.camera_input("Click the button below to take a picture")

if img_file_buffer is not None:
    # Convert the file buffer to a PIL Image
    img = Image.open(img_file_buffer)
    
    # Convert PIL Image to a NumPy array (this is in RGB format)
    frame_rgb = np.array(img)
    
    # FIX #2: Convert RGB to BGR for OpenCV and YOLO
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # --- Perform Inference ---
    # Run YOLO on the BGR frame
    results = model(frame_bgr)
    
    # Use the .plot() method to get the annotated frame (returns a BGR image)
    annotated_frame = results[0].plot()
    
    # --- Display the results ---
    # FIX #1 was caching the model. Now, display the output.
    # We need to convert the BGR image back to RGB for Streamlit's st.image
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    st.image(annotated_frame_rgb, caption="Here are the detected objects!")

    # Optional: Display raw detection data
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        st.write(f"- Detected: **{model.names[class_id]}** with confidence {confidence:.2f}")