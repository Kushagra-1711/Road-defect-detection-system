

# Road Defect Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep_Learning-yellow.svg)

An end-to-end computer vision system for detecting potholes and road defects using both Classical Digital Image Processing (DIP) techniques and Deep Learning (YOLOv8). The project focuses on evaluating performance in real-world, unstructured environments and improving detection accuracy through a hybrid approach.

---

## Project Objectives

* Perform a comparative analysis between Classical DIP techniques and YOLOv8 for pothole detection
* Develop a hybrid pipeline combining deep learning-based localization with contour-based segmentation
* Estimate defect severity using geometric analysis
* Extract GPS metadata from images for location-based mapping

---

## Core Features

* Interactive dashboard for tuning image processing parameters such as Gaussian blur and edge thresholds
* YOLOv8-based detection capable of handling shadows, reflections, and noise
* Hybrid segmentation combining bounding boxes with edge-based contour extraction
* Automatic extraction of EXIF GPS data for mapping
* Severity classification based on detected area

---

## Model Training and Evaluation

### Dataset and Setup

* Dataset: Approximately 3,386 annotated images (Roboflow augmented)
* Model: YOLOv8m
* Training: 25 epochs
* Hardware: Tesla T4 GPU
* Framework: Ultralytics YOLOv8

### Performance Comparison

| Model Version    | Precision | Recall | mAP@50 | mAP@50-95 |
| ---------------- | --------- | ------ | ------ | --------- |
| Baseline YOLOv8m | 0.887     | 0.745  | 0.836  | 0.457     |
| Tuned YOLOv8m    | 0.914     | 0.779  | 0.859  | 0.529     |

### Observations

* Precision improved after hyperparameter tuning, indicating fewer false positives
* Recall increased, suggesting better detection coverage
* mAP@50-95 shows a noticeable improvement, reflecting better localization and generalization

---

## Technical Approach

### Classical DIP Pipeline

* Grayscale conversion
* CLAHE for contrast enhancement
* Gaussian filtering
* Canny edge detection

Limitation: Performance degrades in the presence of noise, shadows, and reflections.

---

### YOLOv8 Detection

* Learns spatial and contextual features
* Provides robust detection under varying environmental conditions
* Outputs bounding boxes for pothole regions

---

### Hybrid Detection Pipeline

1. YOLOv8 identifies regions of interest
2. Regions are cropped from the image
3. Edge detection is applied within the region
4. Contours are extracted to refine boundaries

This approach combines semantic detection with pixel-level precision.

---

## Project Structure

```
app.py                # Streamlit application
preprocessing.py     # Image processing operations
detector.py          # YOLO and hybrid detection logic
analyzer.py          # Severity analysis and EXIF extraction
pothole_yolov8.pt    # Trained model weights
```

---

## Installation and Usage

### Clone the repository

```bash
git clone https://github.com/Kushagra-1711/Real-Time-Pothole-and-Road-Defect-System.git
cd Real-Time-Pothole-and-Road-Defect-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (include **`pothole_yolov8.pt`** in the repo root, ~52 MB).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo, branch **`main`**, main file **`app.py`**.
3. Ensure these files are in the repo:
   - `requirements.txt` (includes `pandas`, `ultralytics`, CPU `torch`)
   - `packages.txt` (Linux libs for OpenCV)
   - `.streamlit/config.toml` (optional upload limit)
4. **First deploy** may take several minutes while PyTorch installs.
5. **YOLO loads on first use** (not at startup) to avoid Cloud memory crashes. DIP mode works even without weights.

If the app still fails, open **Manage app → Logs** and look for `ModuleNotFoundError` or `CUDA out of memory` / `Killed`.

---

## Key Insights

* Classical DIP methods rely on pixel intensity gradients and are sensitive to environmental noise
* YOLOv8 performs better by learning higher-level visual features
* The hybrid approach improves boundary accuracy while maintaining robustness

---

## Limitations

* No temporal analysis for video-based detection
* Environmental factors such as weather are not explicitly modeled
* Severity estimation does not account for depth

---

## Future Work

* Integrate depth estimation for more accurate severity analysis
* Extend to video-based detection and tracking
* Optimize for edge deployment
* Incorporate environmental normalization techniques

---

## Conclusion

The project demonstrates that deep learning significantly outperforms classical methods in real-world conditions. A hybrid approach further enhances performance by combining semantic understanding with precise boundary detection.



