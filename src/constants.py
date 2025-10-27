from pathlib import Path

class BasePaths:
    BASE_DIR = Path("/home/nele_pauline_suffo")
    MODELS_DIR = Path(BASE_DIR/"models")
    DATA_DIR = Path(BASE_DIR/"ProcessedData")
    OUTPUT_DIR = Path(BASE_DIR/"outputs")
    HOME_DIR = Path(BASE_DIR/"projects/naturalistic-social-analysis")
    LOGGING_DIR = Path(OUTPUT_DIR/"dataset_statistics")
    
class DataPaths:
    QUANTEX_VIDEOS_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_videos/") 
    CHILDLENS_VIDEOS_INPUT_DIR = Path(BasePaths.DATA_DIR/"childlens_videos/")
    QUANTEX_IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_videos_processed/")
    CHILDLENS_IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"childlens_videos_processed/")
    ANNO_DIR = Path(BasePaths.DATA_DIR/"quantex_annotations/")
    ANNO_XML_PATH = Path(ANNO_DIR/"annotations.xml")
    ANNO_INDIVIDUAL_DIR = Path(BasePaths.DATA_DIR/"quantex_annotations_individual/")
    ANNO_JSON_PATH = Path(ANNO_DIR/"annotations.json")
    ANNO_DB_PATH = Path(ANNO_DIR/"annotations.db")
    QUANTEX_RAWFRAMES_EXTRACTION_ERROR_LOG = Path(BasePaths.DATA_DIR/"quantex_rawframes_extraction_error.log")
    CHILDLENS_RAWFRAMES_EXTRACTION_ERROR_LOG = Path(BasePaths.DATA_DIR/"childlens_rawframes_extraction_error.log")
    QUANTEX_PROCESSED_VIDEOS_LOG = Path(BasePaths.DATA_DIR/"quantex_processed_videos.log")
    CHILDLENS_PROCESSED_VIDEOS_LOG = Path(BasePaths.DATA_DIR/"childlens_processed_videos.log")
    SUBJECTS_CSV_PATH = Path(BasePaths.DATA_DIR/"age_group.csv")
    INFERENCE_DIR = Path(BasePaths.OUTPUT_DIR/"quantex_inference/")
    INFERENCE_DB_PATH = Path(INFERENCE_DIR/"inference_shorts.db")

class AudioClassification:
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"audio_classification")
    RESULTS_DIR = Path(OUTPUT_DIR/"20251013-144252")
    TRAINED_WEIGHTS_PATH = Path(RESULTS_DIR/"best_model.keras")
    ANNOTATIONS_INPUT_DIR = Path(BasePaths.DATA_DIR/"childlens_annotations/keeper/v1")
    CHILDLENS_PARTICIPANT_INFO = Path(ANNOTATIONS_INPUT_DIR/"childlens_participant_info.csv")
    CHILDLENS_AUDIO_DIR = Path(BasePaths.DATA_DIR/"childlens_audio")
    QUANTEX_AUDIO_DIR = Path(BasePaths.DATA_DIR/"quantex_audio")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"audio_cls_input")
    TRAIN_SEGMENTS_FILE = Path(INPUT_DIR/"train_segments.jsonl")
    VAL_SEGMENTS_FILE = Path(INPUT_DIR/"val_segments.jsonl")
    TEST_SEGMENTS_FILE = Path(INPUT_DIR/"test_segments.jsonl")
    TEST_SECONDS_FILE = Path(INPUT_DIR/"test_segments_per_second.jsonl")
    CACHE_DIR = Path(INPUT_DIR/"feature_cache")
    VTC_RTTM_FILE = Path(OUTPUT_DIR/"all.rttm")
    
class Vocalizations:
    ALICE_OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"audio_word_counts")
    KCHI_OUTPUT_FILE = ALICE_OUTPUT_DIR / "KCHI_output_utterances.txt"
    OTH_OUTPUT_FILE = ALICE_OUTPUT_DIR / "OTH_output_utterances.txt"

class PersonClassification:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.OUTPUT_DIR/'person_classification/resnet18_bilstm_20251008_140718/best.pth')
    EXTRACTION_PROGRESS_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_extraction_progress.txt")
    MISSING_FRAMES_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_missing_frames.txt")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_annotated_rawframes")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"person_classification/")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/yolo_classifications/person_dataset.yaml")
    TRAIN_CSV_PATH = Path(INPUT_DIR/"train.csv")
    VAL_CSV_PATH = Path(INPUT_DIR/"val.csv")
    TEST_CSV_PATH = Path(INPUT_DIR/"test.csv")

class FaceDetection:
    INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"face_detections/")
    TRAINED_WEIGHTS_PATH = Path(OUTPUT_DIR/"yolo12l_20251026_011719/weights/best.pt")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/face/dataset.yaml")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_annotated_rawframes")
    
    PREDICTIONS_JSON_PATH = Path(OUTPUT_DIR/"yolo12l_20251023_144724/yolo12l_validation_20251026_153856/predictions.json")
    RETRAIN_FALSE_POSITIVES_PATH = Path(BasePaths.DATA_DIR/"face_det_input/false_positive_frames.txt")
    DATA_DISTRIBUTION_PATH = Path(BasePaths.OUTPUT_DIR/"dataset_statistics/split_distribution_face_det_20251023_144007.txt")

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
    
class Inference:
    BASE_OUTPUT_DIR = BasePaths.OUTPUT_DIR / "quantex_inference"
    PERSON_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "person_processed.txt"
    FACE_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "face_processed.txt"
    SPEECH_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "speech_processed.txt"
    GROUND_TRUTH_SEGMENTS_CSV = BASE_OUTPUT_DIR / "01_interaction_segments_gt.csv"
    FRAME_LEVEL_INTERACTIONS_CSV = BASE_OUTPUT_DIR / "01_frame_level_social_interactions.csv"
    INTERACTION_SEGMENTS_CSV = BASE_OUTPUT_DIR / "01_interaction_segments.csv"
    UTTERANCE_SEGMENTS_CSV = BASE_OUTPUT_DIR / "01_utterance_segments.csv"
    KCS_SUMMARY_CSV = BASE_OUTPUT_DIR / "02_kcs_summary.csv"
    TURN_TAKING_CSV = BASE_OUTPUT_DIR / "03_turn_taking_summary.csv"
    CDS_SUMMARY_CSV = BASE_OUTPUT_DIR / "04_cds_summary.csv"
    PRESENCE_CSV = BASE_OUTPUT_DIR / "05_presence.csv"
    INTERACTION_COMPOSITION_CSV = BASE_OUTPUT_DIR / "06_interaction_composition.csv"
    
class Evaluation:
    BASE_OUTPUT_DIR = BasePaths.OUTPUT_DIR / "segment_evaluation"
    HYPERPARAMETER_OUTPUT_DIR = BASE_OUTPUT_DIR / "hyperparameter_tuning"
    CONF_MATRIX_COUNTS = BASE_OUTPUT_DIR / "confusion_matrix_counts.png"
    CONF_MATRIX_PERCENTAGES = BASE_OUTPUT_DIR / "confusion_matrix_percentages.png"
    PERFORMANCE_RESULTS_TXT = BASE_OUTPUT_DIR / "performance_results.txt"
    RULE_ABLATION_SUMMARY_CSV = BASE_OUTPUT_DIR / "rule_ablation_summary.csv"
    RULE_ABLATION_PLOT = BASE_OUTPUT_DIR/ "rule_ablation_plot.png"