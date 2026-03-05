import cv2
import numpy as np
import random
import exifread
import io

class DefectAnalyzer:
    def __init__(self):
        pass

    def estimate_severity(self, contours):
        """
        Estimate the severity of each detected pothole based on contour area
        and shape irregularity.
        """
        severities = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Severity logic: Large area or highly irregular shape -> High
            if area > 10000:
                severity = "High"
            elif area > 3000:
                severity = "Medium"
            else:
                severity = "Low"
                
            severities.append(severity)
        return severities

    def extract_location(self, image_bytes):
        """
        Attempt to extract GPS coordinates from image EXIF data.
        If missing, generate a mock set of coordinates for demonstration.
        """
        try:
            tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
            
            # Check for GPS tags
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = self._convert_to_degrees(tags['GPS GPSLatitude'])
                lon = self._convert_to_degrees(tags['GPS GPSLongitude'])
                lat_ref = tags.get('GPS GPSLatitudeRef', 'N').values
                lon_ref = tags.get('GPS GPSLongitudeRef', 'E').values
                
                if lat_ref != 'N':
                    lat = -lat
                if lon_ref != 'E':
                    lon = -lon
                    
                return lat, lon, True
        except Exception:
            pass
            
        # Fallback to simulated location
        lat, lon = self._generate_mock_location()
        return lat, lon, False

    def _convert_to_degrees(self, value):
        d, m, s = value.values
        # Depending on exifread version, these can be Ratio objects
        d = float(d.num) / float(d.den) if hasattr(d, 'num') else float(d)
        m = float(m.num) / float(m.den) if hasattr(m, 'num') else float(m)
        s = float(s.num) / float(s.den) if hasattr(s, 'num') else float(s)
        return d + (m / 60.0) + (s / 3600.0)

    def _generate_mock_location(self):
        # Base coords
        base_lat, base_lon = 28.6139, 77.2090 # New Delhi
        lat_offset = random.uniform(-0.01, 0.01)
        lon_offset = random.uniform(-0.01, 0.01)
        return base_lat + lat_offset, base_lon + lon_offset
