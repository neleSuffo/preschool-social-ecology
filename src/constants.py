from pathlib import Path
from typing import Optional, Tuple

VALID_TARGETS = {"person_face", "all", "person_cls", "face_cls", "gaze_cls", "face_det", "gaze_cls_vit"}

class BasePaths:
    BASE_DIR = Path("/home/nele_pauline_suffo")
    MODELS_DIR = Path(BASE_DIR/"models")
    DATA_DIR = Path(BASE_DIR/"ProcessedData")
    OUTPUT_DIR = Path(BASE_DIR/"outputs")
    HOME_DIR = Path(BASE_DIR/"projects/naturalistic-social-analysis")
    LOGGING_DIR = Path(OUTPUT_DIR/"dataset_statistics")
    
class DataPaths:
    VIDEOS_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_videos/") 
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_videos_processed/")
    ANNO_DIR = Path(BasePaths.DATA_DIR/"quantex_annotations/")
    ANNO_XML_PATH = Path(ANNO_DIR/"annotations.xml")
    ANNO_INDIVIDUAL_DIR = Path(BasePaths.DATA_DIR/"quantex_annotations_individual/")
    ANNO_JSON_PATH = Path(ANNO_DIR/"annotations.json")
    ANNO_DB_PATH = Path(ANNO_DIR/"quantex_annotations.db")
    RAWFRAMES_EXTRACTION_ERROR_LOG = Path(BasePaths.DATA_DIR/"rawframes_extraction_error.log")
    PROCESSED_VIDEOS_LOG = Path(BasePaths.DATA_DIR/"processed_videos.log")

class AudioClassification:
    AUDIO_FILES_DIR = Path(BasePaths.DATA_DIR/"childlens_audio")
    TRAIN_RTTM_FILE = Path(BasePaths.DATA_DIR/"audio_cls_input/train.rttm")
    VAL_RRTM_FILE = Path(BasePaths.DATA_DIR/"audio_cls_input/val.rttm")
    TEST_RTTM_FILE = Path(BasePaths.DATA_DIR/"audio_cls_input/test.rttm")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"audio_classification/runs")

class PersonClassification:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.MODELS_DIR/'yolo11_person_classification.pt')
    EXTRACTION_PROGRESS_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_extraction_progress.txt")
    MISSING_FRAMES_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_missing_frames.txt")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_labels")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"person_classification/")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/yolo_classifications/person_dataset.yaml")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_rawframes_person")

    TRAIN_CSV_PATH = Path(INPUT_DIR/"train.csv")
    VAL_CSV_PATH = Path(INPUT_DIR/"val.csv")
    TEST_CSV_PATH = Path(INPUT_DIR/"test.csv")

    @classmethod
    def get_target_paths(cls, target: str, split_type: str) -> Optional[Tuple[Path, Path]]:
        """
        Get image and label destination paths for a given target and split type.
        
        Parameters
        ----------
        target : str 
            The specific class to get paths for (e.g., 'child_face', 'adult_person', 'gaze')
        split_type : str
            The dataset split ('train', 'val', or 'test')
            
        Returns
        -------
        Optional[Tuple[Path, Path]]
            Tuple of (input_path, output_path) for the requested target class
        """  
        # Add person class paths if target is person-related
        if target in cls.PERSON_CLASSES:
            return (
                cls.INPUT_DIR / split_type / target,
                cls.INPUT_DIR / split_type / target
            )
            
        return None  # Return None if target is not found

class FaceDetection:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.OUTPUT_DIR/"face_detections/20250812_110926_yolo_face/weights/best.pt")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/yolo_detections/face_dataset.yaml")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_labels")
    DATA_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"face_detections/")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_rawframes_face")

    @classmethod
    def get_target_paths(cls, target: str, split_type: str) -> Optional[Tuple[Path, Path]]:
        """
        Get image and label destination paths for a given target and split type.
        
        Parameters
        ----------
        target : str 
            The specific class to get paths for (e.g., 'gaze', 'child_person_face', 'object')
        split_type : str
            The dataset split ('train', 'val', or 'test')
                
        Returns
        -------
        Optional[Tuple[Path, Path]]
            Tuple of (images_path, labels_path) for the requested target class
        
        Raises
        ------
        ValueError
            If split_type is not one of 'train', 'val', 'test'
        """
        # Validate split type
        valid_splits = {'train', 'val', 'test'}
        if split_type not in valid_splits:
            raise ValueError(f"Invalid split_type: {split_type}. Must be one of {valid_splits}")

        # Define path mappings for different targets
        path_mappings = {            
            'face_det': (cls.FACE_DATA_INPUT_DIR / "images" / split_type,
                     cls.FACE_DATA_INPUT_DIR / "labels" / split_type),
        }

        # Return paths if target exists
        if target in path_mappings:
            return path_mappings[target]

        return None

class Proximity:
    REFERENCE_FILE = Path(BasePaths.OUTPUT_DIR/"reference_proximity.json")
    CHILD_CLOSE_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/child_reference_proximity_value_1.jpg")
    CHILD_FAR_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/child_reference_proximity_value_0.jpg")
    ADULT_CLOSE_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/adult_reference_proximity_value_1.jpg")
    ADULT_FAR_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/adult_reference_proximity_value_0.jpg")