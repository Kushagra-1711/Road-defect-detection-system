import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from analyzer import DefectAnalyzer
from detector import PotholeDetector
from preprocessing import ImagePreprocessor

st.set_page_config(page_title="Pothole Detection & Analysis", layout="wide")

st.title("Real-Time Pothole & Road Defect Detection")
st.markdown(
    "Use Digital Image Processing (DIP) or Deep Learning (YOLOv8) to identify potholes."
)


@st.cache_resource
def load_detector():
    return PotholeDetector()


preprocessor = ImagePreprocessor()
detector = load_detector()
analyzer = DefectAnalyzer()

st.sidebar.header("Detection Engine")

yolo_available = detector.has_yolo_weights()

if yolo_available:
    st.sidebar.success("YOLOv8 weights found (loaded on first YOLO inference).")
    detection_mode = st.sidebar.radio(
        "Select Model", ["DIP (Classical)", "YOLOv8 (Deep Learning)"]
    )
    use_yolo = detection_mode == "YOLOv8 (Deep Learning)"
else:
    st.sidebar.warning("YOLOv8 weights not found. Running purely on DIP.")
    st.sidebar.markdown(
        "**For YOLO mode:** commit `pothole_yolov8.pt` to the repo root "
        "(~52 MB) or place it beside `app.py` locally."
    )
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
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr_img is None:
        st.error(
            "Image format error. If this is an iPhone HEIC file renamed as `.jpeg`, "
            "re-save it as JPEG in Paint or Photos, then upload again."
        )
        st.stop()

    img_array = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_array, use_container_width=True)

    with st.spinner("Processing image..."):
        gray, processed_edges = preprocessor.preprocess_pipeline(
            img_array,
            apply_enhancement=apply_clahe,
            apply_noise_reduction=apply_gaussian,
            custom_canny=(canny_low, canny_high),
            blur_kernel=blur_kernel,
        )

        if use_yolo:
            with st.spinner("Loading YOLOv8 and running inference (first run may take ~30s on Cloud)..."):
                contours = detector.detect_potholes(
                    img_array, processed_edges, use_yolo=True
                )
        else:
            contours = detector.detect_potholes(
                img_array, processed_edges, use_yolo=False
            )

        severities = analyzer.estimate_severity(contours)
        output_image = detector.draw_detections(img_array, contours, severities)
        loc_lat, loc_lon, is_real_gps = analyzer.extract_location(uploaded_file.getvalue())

    with col2:
        st.subheader("Detected Potholes")
        st.image(output_image, use_container_width=True)

    st.markdown("---")
    st.header("Analysis Results")

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    high_count = severities.count("High")
    stat_col1.metric("Total Defects Detected", len(contours))
    stat_col2.metric("High Severity Defects", high_count)
    stat_col3.metric("Medium/Low Severity", len(contours) - high_count)

    st.markdown("### Processed Edge Map (Pothole Mask)")
    st.markdown(
        "Binary mask: pothole pixels in black, road background in white."
    )

    height, width = img_array.shape[:2]
    pothole_mask = np.ones((height, width), dtype=np.uint8) * 255
    cv2.drawContours(pothole_mask, contours, -1, 0, -1)

    st.image(
        pothole_mask,
        use_container_width=True,
        caption="Final Extracted Pothole Mask",
    )

    st.markdown("### Location Mapping")
    if is_real_gps:
        st.success(f"**EXIF GPS found:** {loc_lat:.6f}, {loc_lon:.6f}")
    else:
        st.warning(
            f"**No EXIF GPS in this image.** Demo coordinates: {loc_lat:.6f}, {loc_lon:.6f}"
        )

    map_data = pd.DataFrame({"lat": [loc_lat], "lon": [loc_lon]})
    st.map(map_data, zoom=12)
else:
    st.info("Upload a dashcam or pothole image to begin.")
