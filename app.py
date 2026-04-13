import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os

from preprocessing import ImagePreprocessor
from detector import PotholeDetector
from analyzer import DefectAnalyzer

st.set_page_config(page_title="Pothole Detection & Analysis", layout="wide")

st.title("Real-Time Pothole & Road Defect Detection")
st.markdown("Use Digital Image Processing (DIP) or Deep Learning (YOLOv8) to identify potholes.")

# Instantiate components
preprocessor = ImagePreprocessor()
detector = PotholeDetector()
analyzer = DefectAnalyzer()

st.sidebar.header("Detection Engine")

# Check if YOLO is available
yolo_available = detector.yolo_model is not None

if yolo_available:
    st.sidebar.success("YOLOv8 Engine Attached!")
    detection_mode = st.sidebar.radio("Select Model", ["DIP (Classical)", "YOLOv8 (Deep Learning)"])
    use_yolo = (detection_mode == "YOLOv8 (Deep Learning)")
else:
    st.sidebar.warning("YOLOv8 Model not found. Running purely on DIP.")
    st.sidebar.markdown("**Want more accuracy?** Place a trained `pothole_yolov8.pt` in the project folder!")
    st.sidebar.radio("Select Model", ["DIP (Classical)"], disabled=True)
    use_yolo = False

st.sidebar.markdown("---")
st.sidebar.markdown("### DIP Settings")
apply_clahe = st.sidebar.checkbox("Apply CLAHE", value=True)
apply_gaussian = st.sidebar.checkbox("Apply Gaussian Noise Reduction", value=True)
blur_kernel = st.sidebar.slider("Gaussian Blur Kernel", 3, 31, 11, step=2)
st.sidebar.markdown("---")
st.sidebar.subheader("Canny Edge Detection")
canny_low = st.sidebar.slider("Lower Threshold", 0, 255, 50)
canny_high = st.sidebar.slider("Upper Threshold", 0, 255, 150)

uploaded_file = st.file_uploader("Upload a Dashcam Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Decode directly using OpenCV for lightning-fast numpy conversion
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # If OpenCV returns None, the file is structurally flawed (e.g., an Apple HEIC file masquerading as a JPEG)
    if bgr_img is None:
        st.error("Image Format Error! Your file appears to be an Apple HEIC iPhone image that was simply renamed to `.jpeg` without actually being converted. \n\n**To fix this:** Open your image in Windows Paint, click 'File -> Save As -> JPEG', and upload the new file!")
        st.stop()
        
    img_array = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_array, width=700)
        
    with st.spinner("Processing image..."):
        # Run Preprocessing Pipeline
        gray, processed_edges = preprocessor.preprocess_pipeline(
            img_array, 
            apply_enhancement=apply_clahe, 
            apply_noise_reduction=apply_gaussian,
            custom_canny=(canny_low, canny_high),
            blur_kernel=blur_kernel
        )
        
        # Run Detection
        contours = detector.detect_potholes(img_array, processed_edges, use_yolo=use_yolo)
        
        # Run Analysis
        severities = analyzer.estimate_severity(contours)
        
        # Draw Output
        output_image = detector.draw_detections(img_array, contours, severities)
        
        # Extract location
        loc_lat, loc_lon, is_real_gps = analyzer.extract_location(uploaded_file.getvalue())

    with col2:
        st.subheader("Detected Potholes")
        st.image(output_image, width=700)

    # --- UI Layout ---
    # Metrics
    st.markdown("---")
    st.header("Analysis Results")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    high_count = severities.count("High")
    stat_col1.metric("Total Defects Detected", len(contours))
    stat_col2.metric("High Severity Defects", high_count)
    stat_col3.metric("Medium/Low Severity", len(contours) - high_count)
    
    st.markdown("### Processed Edge Map (Pothole Mask)")
    st.markdown("This binary mask isolates the exact pixels of the detected potholes (shown in black) against the rest of the road (shown in white).")
    
    # Generate the strict Black/White Mask as requested
    height, width = img_array.shape[:2]
    pothole_mask = np.ones((height, width), dtype=np.uint8) * 255  # Fill background with white
    cv2.drawContours(pothole_mask, contours, -1, 0, -1)  # Fill detected potholes with black
    
    st.image(pothole_mask, width=700, caption="Final Extracted Pothole Mask")
    
    st.markdown("### Location Mapping")
    if is_real_gps:
        st.success(f"**Precise EXIF GPS Data Found:** {loc_lat:.6f}, {loc_lon:.6f}")
    else:
        st.warning(f"**No Image EXIF GPS Found.** Showing Demonstration Coordinates: {loc_lat:.6f}, {loc_lon:.6f}")
        
    map_data = pd.DataFrame({'lat': [loc_lat], 'lon': [loc_lon]})
    st.map(map_data, zoom=12)
else:
    st.info("Please upload an image from the dataset to begin.")
