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

    def calculate(self, bbox, is_child=True, aspect_ratio_threshold=1):
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect_ratio = width / height if height != 0 else 0

        # Select reference values
        if is_child:
            ref_close, ref_far, ref_ar = self.child_ref_close, self.child_ref_far, self.child_ref_aspect_ratio
        else:
            ref_close, ref_far, ref_ar = self.adult_ref_close, self.adult_ref_far, self.adult_ref_aspect_ratio

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
def calculate_proximities(bboxes, is_child_flags, aspect_ratio_threshold=1):
    """
    Calculate proximity for a list of bounding boxes with age information.

    Args:
        bboxes: list of (x1, y1, x2, y2) tuples
        is_child_flags: list of bools indicating whether each bbox is a child
        aspect_ratio_threshold: max deviation from reference aspect ratio

    Returns:
        list of normalized proximity values
    """
    prox = Proximity()
    proximities = [
        prox.calculate(bbox, is_child=is_child, aspect_ratio_threshold=aspect_ratio_threshold)
        for bbox, is_child in zip(bboxes, is_child_flags)
    ]
    return proximities