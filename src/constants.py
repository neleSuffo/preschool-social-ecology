import os
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
    QUANTEX_PROCESSED_VIDEOS_LOG = Path(BasePaths.DATA_DIR/"quantex_processed_videos.txt")
    CHILDLENS_PROCESSED_VIDEOS_LOG = Path(BasePaths.DATA_DIR/"childlens_processed_videos.txt")
    SUBJECTS_CSV_PATH = Path(BasePaths.DATA_DIR/"age_group.csv")
    INFERENCE_DIR = Path(BasePaths.OUTPUT_DIR/"quantex_inference/")
    INFERENCE_DB_PATH = Path(INFERENCE_DIR/"inference.db")

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
    VTC_RTTM_FILE = Path(OUTPUT_DIR/"vtc_2.0/all.rttm")
    
class Vocalizations:
    ALICE_OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"audio_word_counts")
    KCHI_OUTPUT_FILE = ALICE_OUTPUT_DIR / "KCHI_output_utterances.txt"
    OTH_OUTPUT_FILE = ALICE_OUTPUT_DIR / "OTH_output_utterances.txt"

class PersonClassification:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.MODELS_DIR/"best_yolo_person_cls.pt")
    EXTRACTION_PROGRESS_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_extraction_progress.txt")
    MISSING_FRAMES_FILE_PATH = Path(BasePaths.DATA_DIR/"person_cls_missing_frames.txt")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_annotated_rawframes")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"person_cls_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"person_classification/")
    
    DATA_DISTRIBUTION_PATH = Path(BasePaths.OUTPUT_DIR/"dataset_statistics/split_distribution_person_cls_20251110_114134.txt")

class PersonDetection:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.MODELS_DIR/"best_yolo_person_det.pt")
    EXTRACTION_PROGRESS_FILE_PATH = Path(BasePaths.DATA_DIR/"person_det_extraction_progress.txt")
    MISSING_FRAMES_FILE_PATH = Path(BasePaths.DATA_DIR/"person_det_missing_frames.txt")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"person_det_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_annotated_rawframes")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"person_det_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"person_detection/")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/person/dataset.yaml")

    DATA_DISTRIBUTION_PATH = Path(BasePaths.OUTPUT_DIR/"dataset_statistics/split_distribution_person_det_20251029_202741.txt")

class FaceClassification:
    TRAINED_WEIGHTS_PATH = Path(BasePaths.MODELS_DIR/"best_yolo_face_cls.pt")
    EXTRACTION_PROGRESS_FILE_PATH = Path(BasePaths.DATA_DIR/"face_cls_extraction_progress.txt")
    MISSING_FRAMES_FILE_PATH = Path(BasePaths.DATA_DIR/"face_cls_missing_frames.txt")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_cls_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_annotated_rawframes")
    INPUT_DIR = Path(BasePaths.DATA_DIR/"face_cls_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"face_classifications/")
    DATA_DISTRIBUTION_PATH = Path(BasePaths.OUTPUT_DIR/"dataset_statistics/split_distribution_face_cls_20251110_114134.txt")

class FaceDetection:
    FACE_MODEL_PATH = Path(BasePaths.MODELS_DIR/"yolov12l-face.pt") # pretrained face detection model from ultralytics hub
    INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_input")
    OUTPUT_DIR = Path(BasePaths.OUTPUT_DIR/"face_detections/")
    TRAINED_WEIGHTS_PATH = Path(BasePaths.MODELS_DIR/"best_yolo_face_det.pt")
    DATA_CONFIG_PATH = Path(BasePaths.HOME_DIR/"src/models/face/dataset.yaml")
    LABELS_INPUT_DIR = Path(BasePaths.DATA_DIR/"face_det_labels")
    IMAGES_INPUT_DIR = Path(BasePaths.DATA_DIR/"quantex_annotated_rawframes")
    
    PREDICTIONS_JSON_PATH = Path(OUTPUT_DIR/"yolo12l_20251023_144724/yolo12l_validation_20251026_153856/predictions.json")
    RETRAIN_FALSE_POSITIVES_PATH = Path(BasePaths.DATA_DIR/"face_det_input/false_positive_frames.txt")
    DATA_DISTRIBUTION_PATH = Path(BasePaths.OUTPUT_DIR/"dataset_statistics/split_distribution_face_det_20251026_155901.txt")

class Proximity:
    REFERENCE_VALUES = {"child_ref_close": 458185,
                        "child_ref_far": 308,
                        "adult_ref_close": 442980,
                        "adult_ref_far": 208,
                        "child_ref_aspect_ratio": 0.965166908563135,
                        "adult_ref_aspect_ratio": 0.6461352657004831
                        }
    
class Inference:
    PERSON_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "person_processed.txt"
    FACE_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "face_processed.txt"
    SPEECH_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "speech_processed.txt"
    BOOK_LOG_FILE_PATH = BasePaths.OUTPUT_DIR / "logs" / "book_processed.txt"
    
class Analysis:
    QUANTEX_VIDEOS_LIST_FILE = Path(BasePaths.DATA_DIR/"quantex_video_list_inference_tuning.txt") # list of all quantex videos for inference
    BASE_OUTPUT_DIR = BasePaths.OUTPUT_DIR / "quantex_analysis"
    CONF_MATRIX_COUNTS = Path("confusion_matrix_counts.png")
    CONF_MATRIX_PERCENTAGES = Path("confusion_matrix_percentages.png")
    PERFORMANCE_RESULTS_TXT = Path("performance_results.txt")
    RULE_ABLATION_SUMMARY_CSV = BASE_OUTPUT_DIR / "rule_ablation_summary.csv"
    RULE_ABLATION_PLOT = BASE_OUTPUT_DIR/ "rule_ablation_plot.png"
    GROUND_TRUTH_SEGMENTS_CSV = BASE_OUTPUT_DIR / "01_interaction_segments_gt.csv"
    GT_1_FILE_PATH = BASE_OUTPUT_DIR / "interaction_segments_clara.csv"
    GT_2_FILE_PATH = BASE_OUTPUT_DIR / "interaction_segments_lotta.csv"
    GT_1_SECONDWISE_FILE_PATH = BASE_OUTPUT_DIR / "gt_ann1_secondwise.csv"
    GT_2_SECONDWISE_FILE_PATH = BASE_OUTPUT_DIR / "gt_ann2_secondwise.csv"
    PRED_SECONDWISE_FILE_PATH = BASE_OUTPUT_DIR / "pred_secondwise.csv"
    
    # -- Dynamic Folder Logic --
    # 1. Define the default fallback
    DEFAULT_RUN_NAME = "analysis_20260410_202054_tertiary_final"
    # 2. Look for the environment variable 'RUN_FOLDER_NAME'
    # os.getenv returns the string from the system, or the default if not found
    active_run_name = os.getenv("RUN_FOLDER_NAME", DEFAULT_RUN_NAME)
    # 3. Build the folder path
    FINAL_OUTPUT_FOLDER = BASE_OUTPUT_DIR / active_run_name
    
    FRAME_ANALYSIS_SCRIPT = Path("heuristics/pipeline_frame_level_analysis.py")
    SEGMENT_CREATION_SCRIPT = Path("heuristics/pipeline_video_level_analysis.py")
    FRAME_LEVEL_INTERACTIONS_CSV = Path(FINAL_OUTPUT_FOLDER/"frame_level_social_interactions.csv")
    INTERACTION_SEGMENTS_CSV = Path(FINAL_OUTPUT_FOLDER/"interaction_segments.csv")
    #UTTERANCE_SEGMENTS_CSV = FINAL_OUTPUT_FOLDER / "00_utterance_segments.csv"
    KCS_SUMMARY_CSV = FINAL_OUTPUT_FOLDER / "02_kcs_summary.csv"
    GLOBAL_KCS_SUMMARY_CSV = FINAL_OUTPUT_FOLDER / "02a_global_kcs_summary.csv"
    CDS_SUMMARY_CSV = FINAL_OUTPUT_FOLDER / "03_cds_summary.csv"
    GLOBAL_CDS_SUMMARY_CSV = FINAL_OUTPUT_FOLDER / "03a_global_cds_summary.csv"
    TURN_TAKING_CSV = FINAL_OUTPUT_FOLDER / "04_turn_taking_summary.csv"
    GLOBAL_TURN_TAKING_CSV = FINAL_OUTPUT_FOLDER / "04a_global_turn_taking_summary.csv"
    TURN_DURATION_CSV = FINAL_OUTPUT_FOLDER / "04b_turn_durations_summary.csv"
    INTERACTION_COMPOSITION_CSV = FINAL_OUTPUT_FOLDER / "05_interaction_composition.csv"
    TEMP_CUT_FACE_DIR = BASE_OUTPUT_DIR / "temp_cut_faces"
    
class Visualization:
    FACE_HEATMAP_PATH = Analysis.FINAL_OUTPUT_FOLDER / "face_heatmaps_per_state.png"
    HEATMAP_SUBFOLDER = "heatmaps"
    FULL_MATCHED_FACES_DETECTIONS = Analysis.FINAL_OUTPUT_FOLDER / HEATMAP_SUBFOLDER / "full_matched_faces_detections.csv"
    GT_MATCHED_FACES_DETECTIONS = Analysis.FINAL_OUTPUT_FOLDER / HEATMAP_SUBFOLDER / "gt_matched_faces_detections.csv"