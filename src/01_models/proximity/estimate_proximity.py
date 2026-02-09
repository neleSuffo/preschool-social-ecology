import logging
import numpy as np
from constants import Proximity as ProximityConstants

class Proximity:
    def __init__(self):
        (
            self.child_ref_close,
            self.child_ref_far,
            self.adult_ref_close,
            self.adult_ref_far,
            self.child_ref_aspect_ratio,
            self.adult_ref_aspect_ratio
        ) = self._load_stored_references()

    def _load_stored_references(self):
        return (
            ProximityConstants.REFERENCE_VALUES["child_ref_close"],
            ProximityConstants.REFERENCE_VALUES["child_ref_far"],
            ProximityConstants.REFERENCE_VALUES["adult_ref_close"],
            ProximityConstants.REFERENCE_VALUES["adult_ref_far"],
            ProximityConstants.REFERENCE_VALUES["child_ref_aspect_ratio"],
            ProximityConstants.REFERENCE_VALUES["adult_ref_aspect_ratio"]
        )

    def calculate(self, bbox, class_id, aspect_ratio_threshold=1) -> float:
        """
        Calculate proximity based on bounding box and class ID.
        
        Parameters:
        ----------
        bbox: list or tuple of (x1, y1, x2, y2)
            Bounding box coordinates.
        class_id: int
            Class ID (0 for child, 1 for adult).
        aspect_ratio_threshold: float
            Maximum allowed deviation from reference aspect ratio.
            
        Returns:
        -------
        float: 
            Normalized proximity value between 0.0 and 1.0.
        """
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect_ratio = width / height if height != 0 else 0

        # Select reference values
        if class_id == 0:  # Child
            ref_close, ref_far, ref_ar = self.child_ref_close, self.child_ref_far, self.child_ref_aspect_ratio
        elif class_id == 1:
            ref_close, ref_far, ref_ar = self.adult_ref_close, self.adult_ref_far, self.adult_ref_aspect_ratio
        else:
            logging.error(f"Invalid class_id: {class_id}. Must be 0 (child) or 1 (adult).")
            return None
        
        # Partial face check
        if abs(aspect_ratio - ref_ar) > aspect_ratio_threshold:
            logging.debug("Partial face detected - returning maximum proximity")
            return 1.0

        # Edge cases
        if area >= ref_close:
            return 1.0
        if area <= ref_far:
            return 0.0

        # Logarithmic normalization
        proximity = (np.log(area) - np.log(ref_far)) / (np.log(ref_close) - np.log(ref_far))
        return np.clip(proximity, 0.0, 1.0)


# --- Main function for external calls ---
def calculate_proximity(bbox, class_id, aspect_ratio_threshold=1):
    """
    Calculate proximity for a single bounding box with age information.

    Args:
        bbox: (x1, y1, x2, y2) tuple or list
        class_id: class ID (0 for child, 1 for adult)
        aspect_ratio_threshold: max deviation from reference aspect ratio

    Returns:
        float: normalized proximity value
    """
    prox = Proximity()
    proximity = prox.calculate(bbox, class_id=class_id, aspect_ratio_threshold=aspect_ratio_threshold)
    return proximity