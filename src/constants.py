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
    SUBJECTS_CSV_PATH = Path(BasePaths.DATA_DIR/"quantex_subjects.csv")
    INFERENCE_DB_PATH = Path(BasePaths.OUTPUT_DIR/"quantex_inference/inference_database.db")

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
    TRAIN_CSV_PATH = Path(INPUT_DIR/"train.csv")
    VAL_CSV_PATH = Path(INPUT_DIR/"val.csv")
    TEST_CSV_PATH = Path(INPUT_DIR/"test.csv")

class FaceDetection:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.OUTPUT_DIR/"face_detections/yolo12m_20250814_112743/weights/best.pt")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/face/dataset.yaml")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_labels")
    DATA_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"face_detections/")

class Proximity:
    REFERENCE_VALUES = {"child_ref_close": 458185,
                        "child_ref_far": 308,
                        "adult_ref_close": 442980,
                        "adult_ref_far": 208,
                        "child_ref_aspect_ratio": 0.965166908563135,
                        "adult_ref_aspect_ratio": 0.6461352657004831
                        }
    CHILD_CLOSE_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/child_reference_proximity_value_1.jpg")
    CHILD_FAR_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/child_reference_proximity_value_0.jpg")
    ADULT_CLOSE_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/adult_reference_proximity_value_1.jpg")
    ADULT_FAR_IMAGE_PATH = Path(BasePaths.OUTPUT_DIR/"proximity_sampled_frames/adult_reference_proximity_value_0.jpg")