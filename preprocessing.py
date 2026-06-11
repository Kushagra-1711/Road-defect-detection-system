import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        pass

    def to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def enhance_contrast(self, image):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def reduce_noise(self, image, kernel_size=11):
        # A stronger blur eliminates the rough gravel texture so Canny only detects large defects
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def detect_edges(self, image, low_thresh=50, high_thresh=150):
        # Apply Canny Edge Detection
        return cv2.Canny(image, low_thresh, high_thresh)

    def morphological_operations(self, edges):
        # Dialed back from extreme dilation to prevent turning the image into solid white
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed

    def preprocess_pipeline(self, image, apply_enhancement=True, apply_noise_reduction=True, custom_canny=None, blur_kernel=11):
        gray = self.to_grayscale(image)
        if apply_enhancement:
            gray = self.enhance_contrast(gray)
        if apply_noise_reduction:
            gray = self.reduce_noise(gray, kernel_size=blur_kernel)
            
        if custom_canny:
            edges = self.detect_edges(gray, custom_canny[0], custom_canny[1])
        else:
            edges = self.detect_edges(gray)
            
        processed_edges = self.morphological_operations(edges)
        return gray, processed_edges
