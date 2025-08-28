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
    ANNO_DB_PATH = Path(ANNO_DIR/"annotations.db")
    RAWFRAMES_EXTRACTION_ERROR_LOG = Path(BasePaths.DATA_DIR/"rawframes_extraction_error.log")
    PROCESSED_VIDEOS_LOG = Path(BasePaths.DATA_DIR/"processed_videos.log")
    SUBJECTS_CSV_PATH = Path(BasePaths.DATA_DIR/"age_group.csv")
    INFERENCE_DIR = Path(BasePaths.OUTPUT_DIR/"quantex_inference/")
    INFERENCE_DB_PATH = Path(INFERENCE_DIR/"inference.db")

class AudioClassification:
    RESULTS_DIR = Path(BasePaths.OUTPUT_DIR/"audio_classification/runs/20250703-165356")
    ANNOTATIONS_INPUT_DIR = Path(BasePaths.DATA_DIR/"childlens_annotations/keeper/v1")
    CHILDLENS_PARTICIPANT_INFO = Path(ANNOTATIONS_INPUT_DIR/"childlens_participant_info.csv")
    AUDIO_FILES_DIR = Path(BasePaths.DATA_DIR/"childlens_audio")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"audio_cls_input")
    TRAIN_RTTM_FILE = Path(INPUT_DIR/"train.rttm")
    VAL_RTTM_FILE = Path(INPUT_DIR/"dev.rttm")
    TEST_RTTM_FILE = Path(INPUT_DIR/"test.rttm")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"audio_classification/runs")
    TRAIN_SEGMENTS_FILE = Path(INPUT_DIR/"train_segments_w3p0_s1p0.jsonl")
    VAL_SEGMENTS_FILE = Path(INPUT_DIR/"val_segments_w3p0_s1p0.jsonl")
    TEST_SEGMENTS_FILE = Path(INPUT_DIR/"test_segments_w3p0_s1p0.jsonl")

class Vocalizations:
    ALICE_OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"audio_word_counts")
    KCHI_OUTPUT_FILE = ALICE_OUTPUT_DIR / "KCHI_output_utterances.txt"
    #OTH_OUTPUT_FILE = ALICE_OUTPUT_DIR / "FEM_MAL_output_utterances.txt"
    OTH_OUTPUT_FILE = ALICE_OUTPUT_DIR / "OTH_output_utterances.txt"

class PersonClassification:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.OUTPUT_DIR/'person_classification/resnet18_bilstm_20250815_221214/best.pth')
    EXTRACTION_PROGRESS_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_extraction_progress.txt")
    MISSING_FRAMES_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_missing_frames.txt")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_rawframes_cvat")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"person_classification/")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/yolo_classifications/person_dataset.yaml")
    TRAIN_CSV_PATH = Path(INPUT_DIR/"train.csv")
    VAL_CSV_PATH = Path(INPUT_DIR/"val.csv")
    TEST_CSV_PATH = Path(INPUT_DIR/"test.csv")

class FaceDetection:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.OUTPUT_DIR/"face_detections/yolo12l_20250821_155703/weights/best.pt")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/face/dataset.yaml")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_rawframes_cvat")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_input")
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
    
class ResearchQuestions:
    OUTPUT_BASE_DIR = Path("/home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/results")
    RQ1_OUTPUT_DIR = OUTPUT_BASE_DIR / "rq_01"
    FRAME_LEVEL_INTERACTIONS_CSV = RQ1_OUTPUT_DIR / "frame_level_social_interactions.csv"
    INTERACTION_SEGMENTS_CSV = RQ1_OUTPUT_DIR / "interaction_segments.csv"
    CHILD_ID_CSV = RQ1_OUTPUT_DIR / "child_id_age_mapping.csv"
    
    RQ_2_OUTPUT_DIR = OUTPUT_BASE_DIR / "rq_02"
    UTTERANCE_SEGMENTS_CSV = RQ_2_OUTPUT_DIR / "utterance_segments.csv"
    WORD_SUMMARY_CSV = RQ_2_OUTPUT_DIR / "word_summary.csv"

    RQ_3_OUTPUT_DIR = OUTPUT_BASE_DIR / "rq_03"
    
    
    RQ_4_OUTPUT_DIR = OUTPUT_BASE_DIR / "rq_04" 
    PRESENCE_COUNTS_CSV = RQ_4_OUTPUT_DIR / "presence_counts.csv"
    
    RQ5_OUTPUT_DIR = OUTPUT_BASE_DIR / "rq_05"