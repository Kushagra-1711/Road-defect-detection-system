# 🛣️ Real-Time Pothole & Road Defect Detection
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep_Learning-yellow.svg)

An interactive, end-to-end computer vision pipeline designed for Smart Transportation & Urban Infrastructure Monitoring. This project provides a comparative analysis framework evaluating the efficacy of Classical Digital Image Processing (DIP) techniques against modern spatial Deep Learning neural networks (YOLOv8) when tackling heavily unstructured real-world anomalies like road deterioration.

## 🎯 Project Objectives
1. **DIP vs Deep Learning Study:** Observe how strict pixel-based algorithmic rules (Canny Edges, CLAHE) naturally break apart handling chaotic textures (gravel, water reflections, extreme shadows) compared to AI semantic concept mapping.
2. **Hybrid Semantic Extraction:** Create a novel custom pipeline that leverages YOLOv8 bounding boxes as a regional constraint to isolate and extract a pixel-perfect jagged pothole mask via the classical edge map!
3. **Automated Severity Queue:** Grade infrastructural hazards based on mathematically calculated geometric surface areas to build a city planning queue.
4. **Geo-Spatial Tracing:** Seamlessly unpack hidden image EXIF layers to harvest GPS coordinates for Map routing, with robust localized fallbacks.

## ✨ Core Features
*   🎛️ **Live Parameter Tuning Dashboard:** Instantly adjust Gaussian Blurs and Edge Threshold sliders in real-time to watch mathematical matrices respond to the image live.
*   🤖 **YOLOv8 Semantic Brain:** An integrated neural network capable of perfectly isolating craters through water and shadows without requiring manual filtering.
*   🗺️ **EXIF Geo-Location Engine:** Robustly detects and maps exact geographical coordinates from mobile phone photos directly onto an interactive dashboard map.
*   📊 **Threat Severity Analytics:** Automatically categorizes surface damages into Low, Medium, or High severities using boundary box scaling.

## ⚙️ Installation & Quickstart

**1. Clone the repository:**
```bash
git clone https://github.com/Kushagra-1711/Real-Time-Pothole-and-Road-Defect-System.git
cd Real-Time-Pothole-and-Road-Defect-System
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the interactive dashboard:**
```bash
streamlit run app.py
```
*(The dashboard will output a local network URL, typically `http://localhost:8501`, which will open automatically in your browser.)*

## 📂 Project Architecture

*   `app.py`: The main runtime engine orchestrating the Streamlit Web Application and routing visual widgets.
*   `preprocessing.py`: The Classical DIP backbone handling Grayscaling, CLAHE contrast scaling, Gaussian filtering, and Canny Edge detection.
*   `detector.py`: The Hybrid Segmentation engine. Houses the logic that switches between pure DIP algorithms and YOLOv8 inference, alongside the exact Pixel-Contour mask extractor.
*   `analyzer.py`: The geometric mathematics model containing polygon severity grading and EXIF binary unpacking logic for the GPS map.
*   `pothole_yolov8.pt`: The pre-trained Deep Learning PyTorch weights file for the YOLO neural network model.

## 🧠 Technical Learnings
During development, a key takeaway discovered was the rigidity of **Classical DIP** in unconstrained real-world environments. When an uploaded pothole image contains water, the stark reflection acts as an optical camouflage; it fractures the pixel gradients, causing edge-tracing algorithms to completely miss the target or classify harmless pebbles as threats. To resolve this, **YOLOv8 (Spatial Deep Learning)** proved inherently superior by recognizing the broader semantic visual *context* of a pothole, completely ignoring the noise from the water reflections.
