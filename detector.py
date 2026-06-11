import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class PotholeDetector:
    def __init__(self, yolo_model_path="pothole_yolov8.pt"):
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None

    def has_yolo_weights(self):
        return YOLO is not None and os.path.exists(self.yolo_model_path)

    @property
    def yolo_model(self):
        """Load YOLO weights on first use (keeps Streamlit Cloud startup light)."""
        if self._yolo_model is None and self.has_yolo_weights():
            self._yolo_model = YOLO(self.yolo_model_path)
        return self._yolo_model

    def detect_potholes(self, original_img, processed_edges, use_yolo=False):
        contours = []
        
        if use_yolo and self.has_yolo_weights():
            model = self.yolo_model
            if model is None:
                return contours
            # YOLOv8 Detection
            results = model(original_img)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Hybrid Approach: Extract EXACT jagged pixel boundary using Edge map crop!
                roi_edges = processed_edges[y1:y2, x1:x2]
                roi_contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if roi_contours:
                    # Snag the largest edge trace inside the YOLO box
                    largest_roi_cnt = max(roi_contours, key=cv2.contourArea)
                    # Shift the exact contour coordinates back to the main image space
                    extracted_contour = largest_roi_cnt + np.array([[[x1, y1]]])
                    contours.append(extracted_contour)
                else:
                    # Fallback to rectangle if edge map is empty inside the box
                    contours.append(np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]]))
            return contours
            
        # --- DIP Contour Detection Logic ---
        base_contours, _ = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = original_img.shape[:2]
        
        for cnt in base_contours:
            # Ignore sky/trees (top 35% of image)
            x, y, w, h = cv2.boundingRect(cnt)
            if y < height * 0.35:
                continue

            area = w * h
            if 500 < area < (height * width * 0.8):
                # We do NOT use convexHull here anymore; we strictly preserve the exact raw natural edge!
                contours.append(cnt)

        return contours

    def draw_detections(self, image, contours, severities=None):
        out_img = image.copy()
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            color = (0, 255, 255) 
            label = "Defect"
            if severities and i < len(severities):
                sev = severities[i]
                if sev == "High":
                    color = (255, 0, 0)
                elif sev == "Medium":
                    color = (255, 165, 0)
                else: 
                    color = (0, 255, 0)
                label = f"{sev} Severity"

            cv2.rectangle(out_img, (x, y), (x+w, y+h), color, 2)
            cv2.drawContours(out_img, [cnt], -1, color, 1)
            cv2.putText(out_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        return out_img
