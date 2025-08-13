import cv2
import logging
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from constants import Proximity, DetectionPaths

logging.basicConfig(level=logging.INFO)

class ProximityCalculator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_references()
        return cls._instance
    
    def _load_references(self):
        """Load or compute reference metrics once"""
        self.ref_metrics = self._load_or_compute_references()
        
    def _load_or_compute_references(self):
        stored = self._load_stored_references()
        if stored: 
            return stored
        return self._compute_new_references()

    def _load_stored_references(self):
        try:
            with open(Proximity.reference_file, "r") as f:
                metrics = json.load(f)
            required_keys = [
                "child_ref_close", "child_ref_far",
                "adult_ref_close", "adult_ref_far",
                "child_ref_aspect_ratio", "adult_ref_aspect_ratio"
            ]
            if all(k in metrics for k in required_keys):
                return (
                    metrics["child_ref_close"], metrics["child_ref_far"],
                    metrics["adult_ref_close"], metrics["adult_ref_far"],
                    metrics["child_ref_aspect_ratio"], metrics["adult_ref_aspect_ratio"]
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Couldn't load stored references: {str(e)}")
        return None

    def _compute_new_references(self):
        model = YOLO(DetectionPaths.person_face_trained_weights_path)
        
        def process_face_boxes(image_path):
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load reference image: {image_path}")
            results = model(img)
            return [
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]), 
                 int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                for r in results for box in r.boxes 
                if int(box.cls) in {2, 3}  # Face classes only
            ]

        def calculate_face_metrics(faces, label=""):
            if not faces:
                raise ValueError(f"No faces detected in {label} reference image")
            
            areas = []
            aspect_ratios = []
            
            for (x1, y1, x2, y2) in faces:
                w, h = x2 - x1, y2 - y1
                areas.append(w * h)
                aspect_ratios.append(w / h if h != 0 else 0)
            
            return {
                "max_area": max(areas),
                "min_area": min(areas),
                "avg_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0.75
            }

        # Process all reference images
        try:
            child_close_faces = process_face_boxes(Proximity.child_close_image_path)
            child_far_faces = process_face_boxes(Proximity.child_far_image_path)
            adult_close_faces = process_face_boxes(Proximity.adult_close_image_path)
            adult_far_faces = process_face_boxes(Proximity.adult_far_image_path)
            
            child_close_metrics = calculate_face_metrics(child_close_faces, "child close")
            child_far_metrics = calculate_face_metrics(child_far_faces, "child far")
            adult_close_metrics = calculate_face_metrics(adult_close_faces, "adult close")
            adult_far_metrics = calculate_face_metrics(adult_far_faces, "adult far")
            
            metrics = {
                "child_ref_close": child_close_metrics["max_area"],
                "child_ref_far": child_far_metrics["min_area"],
                "adult_ref_close": adult_close_metrics["max_area"],
                "adult_ref_far": adult_far_metrics["min_area"],
                "child_ref_aspect_ratio": child_close_metrics["avg_aspect_ratio"],
                "adult_ref_aspect_ratio": adult_close_metrics["avg_aspect_ratio"]
            }
            
            with open(Proximity.reference_file, "w") as f:
                json.dump(metrics, f)
                
            return (
                metrics["child_ref_close"], metrics["child_ref_far"],
                metrics["adult_ref_close"], metrics["adult_ref_far"],
                metrics["child_ref_aspect_ratio"], metrics["adult_ref_aspect_ratio"]
            )
            
        except Exception as e:
            logging.error(f"Failed to compute references: {str(e)}")
            raise

    def calculate(self, bbox, is_child=True, aspect_ratio_threshold=1):
        """
        Calculate proximity for a face bounding box
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates
            is_child: Whether the face is a child (vs adult)
            aspect_ratio_threshold: Max deviation from reference aspect ratio
            
        Returns:
            Normalized proximity value (0-1)
        """
        if not self.ref_metrics:
            raise ValueError("Reference metrics not available")
            
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect_ratio = width / height if height != 0 else 0
        
        # Get appropriate reference values
        if is_child:
            ref_close, ref_far, ref_ar = self.ref_metrics[0], self.ref_metrics[1], self.ref_metrics[4]
        else:
            ref_close, ref_far, ref_ar = self.ref_metrics[2], self.ref_metrics[3], self.ref_metrics[5]
        
        # Check for partial face based on aspect ratio
        if abs(aspect_ratio - ref_ar) > aspect_ratio_threshold:
            logging.debug("Partial face detected - returning maximum proximity")
            return 1.0
        
        # Handle edge cases
        if area >= ref_close:
            return 1.0
        if area <= ref_far:
            return 0.0
            
        # Normalize between reference values using logarithmic scale
        proximity = (np.log(area) - np.log(ref_far)) / (np.log(ref_close) - np.log(ref_far))
        return np.clip(proximity, 0.0, 1.0)
