from collections import defaultdict
from dynaconf import Dynaconf
from pathlib import Path

# Dynaconf settings
SETTINGS = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

# Whether to generate the output video with the detection results
GENERATE_DETECTION_OUTPUT_VIDEO = SETTINGS.get("generate_detection_output_video", True)

# General Preprocessing and Data Configuration
class DataConfig:
    """General configuration for data processing and labels."""
    VIDEO_FILE_EXTENSION = ".mp4"
    FRAME_STEP_INTERVAL = 10
    EXTRACTION_FPS = 1
    TRAIN_SPLIT_RATIO = 0.7
    RANDOM_SEED = 42
    FPS = 30
    FRAME_WIDTH = 2304
    FRAME_HEIGHT = 1296
    VIDEO_BATCH_SIZE = 16
    VALID_EXTENSIONS = [".jpg", ".PNG"]
    EXCLUDED_VIDEOS = [
        'quantex_at_home_id260275_2022_05_27_01.mp4',
        'quantex_at_home_id260275_2022_04_16_01.mp4', 
        'quantex_at_home_id260275_2022_04_12_01.mp4',
        'quantex_at_home_id258704_2022_05_07_03.mp4',
        'quantex_at_home_id258704_2022_05_07_04.mp4',
        'quantex_at_home_id258704_2022_05_10_02.mp4',
        'quantex_at_home_id258704_2022_05_15_01.mp4',
        'quantex_at_home_id262565_2022_05_26_03.mp4',
    ]   

class PipelineConfig:
    """Configuration for the overall detection pipeline."""
    VIDEOS_TO_NOT_PROCESS = [
        "quantex_at_home_id260694_2022_06_29_01.MP4",
        "quantex_at_home_id260694_2022_06_29_02.MP4",
        "quantex_at_home_id260694_2022_06_29_03.MP4",
        "quantex_at_home_id260694_2022_05_21_01.MP4",
        "quantex_at_home_id260694_2022_05_21_02.MP4",
        "quantex_at_home_id254922_2022_06_29_01.MP4",
        "quantex_at_home_id254922_2022_06_29_02.MP4",
        "quantex_at_home_id254922_2022_06_29_03.MP4",
    ]

class LabelMapping:
    """Mappings for labels, IDs, and supercategories."""
    LABEL_TO_ID_MAPPING = defaultdict(
        lambda: 99,
        {
            "person": 1,
            "reflection": 2,
            "book": 3,
            "animal": 4,
            "toy": 5,
            "kitchenware": 6,
            "screen": 7,
            "food": 8,
            "object": 9,
            "other_object": 12,
            "face": 10,
            'child_body_parts': 11,
            "voice": 20,
            "noise": -1,
        },
    )

    ID_TO_SUPERCATEGORY_MAPPING = defaultdict(
        lambda: "unknown",
        {
            1: "person",
            2: "reflection",
            3: "object",
            4: "object",
            5: "object",
            6: "object",
            7: "object",
            8: "object",
            9: "object",
            12: "object",
            10: "face",
            20: "voice",
            -1: "noise",
            99: "unknown",
        },
    )
    unknown_label_id = -1
    unknown_supercategory = "unknown"

# Specific Task Configurations
class PersonConfig:
    """Configuration for person detection and classification."""
    MODEL_NAME = "resnet18_bilstm"
    # Ratio of training data to use for training
    TRAIN_SPLIT_RATIO = 0.6
    # Ratio of class-to-class samples in each dataset split
    MAX_CLASS_RATIO_THRESHOLD = 0.9
    AGE_GROUP_TO_CLASS_ID = {
        'Inf': 0,
        'Child': 0,
        'Teen': 1,
        'Adult': 1,
    }
    MODEL_CLASS_ID_TO_LABEL = {
        0: "child",
        1: "adult"
    }
    DATABASE_CATEGORY_IDS = [1, 2]
    TARGET_LABELS = ['child', 'adult']
    
    NUM_EPOCHS = 100
    # number of videos per batch
    BATCH_SIZE = 32
    # maximum sequence length for RNN
    MAX_SEQ_LEN = 10
    LR = 1e-4
    FREEZE_CNN = True
    PATIENCE = 15
    SEQUENCE_LENGTH = 60
    DROPOUT = 0.5
    WEIGHT_DECAY = 1e-5
    FEAT_DIM = 512
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    BIDIRECTIONAL = True
    NUM_OUTPUTS = 2
    BACKBONE = 'resnet18'
    CONFIDENCE_THRESHOLD = 0.5
    BATCH_SIZE_INFERENCE = 64
    WINDOW_SIZE = 60
    STRIDE = 30
    MODEL_ID = 2

class FaceConfig:
    """Configuration for face detection and classification."""
    MODEL_SIZE = 'm'  # Default model size
    MODEL_NAME = f"yolo12{MODEL_SIZE}"
    AGE_GROUP_TO_CLASS_ID = {
        'infant': 0,
        'child': 0,
        'teen': 1,
        'adult': 1,
    }
    MODEL_CLASS_ID_TO_LABEL = {
        0: "child",
        1: "adult"
    }
    DATABASE_CATEGORY_IDS = [10]
    TARGET_LABELS = ['child', 'adult']

    TRAIN_SPLIT_RATIO = 0.6
    MAX_CLASS_RATIO_THRESHOLD = 0.60
    NUM_EPOCHS = 300
    BATCH_SIZE = 20
    IMG_SIZE = 832
    LR = 1e-4
    MODEL_ID = 1

class AudioConfig:
    """Configuration for audio classification."""
    VALID_RTTM_CLASSES = ['OHS', 'CDS', 'KCHI']
    VALID_EVENT_IDS = {"child_talking", "other_person_talking", "overheard_speech", "singing/humming"}
    SR = 16000
    N_MELS = 256
    HOP_LENGTH = 512
    WINDOW_DURATION = 3.0
    WINDOW_STEP = 1.0
    EPOCHS = 100
    MODEL_ID = 3

class KchiVoc_Config:
    """Configuration for KCHI vocalizations."""
    MODEL_ID = 4

# Yolo Model and Training Configurations
class YoloConfig:
    """YOLO model and training parameters."""
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    IOU_THRESHOLD = 0.35
    IMG_SIZE = (320, 640)
    BEST_IOU = 0.5
    DETECTION_MAPPING = {
        0: 'infant/child',
        1: 'adult',
        2: 'infant/child face',
        3: 'adult face',
        4: 'child body parts',
        5: 'book',
        6: 'toy',
        7: 'kitchenware',
        8: 'screen',
        9: 'food',
        10: 'other_object'
    }
    ALL_TARGET_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    PERSON_FACE_TARGET_CLASS_IDS = [1, 2, 10, 11]
    OBJECT_TARGET_CLASS_IDS = [3, 4, 5, 6, 7, 8, 12]
    ALL_DET = {
        1: 0, 2: 0, 10: 1, 11: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 12: 3,
    }
    ALL_INSTANCES = {
        (1, 'inf'): 0, (1, 'child'): 0, (1, 'teen'): 1, (10, 'adult'): 1,
        (2, 'inf'): 0, (2, 'child'): 0, (2, 'teen'): 1, (2, 'adult'): 1,
        (10, 'infant'): 2, (10, 'child'): 2, (10, 'teen'): 3, (10, 'adult'): 3,
        11: 4, 3: 5, 4: 11, 5: 6, 6: 7, 7: 8, 8: 9, 12: 10
    }
    OBJECT_DET = {
        (3, 'Yes'): 0, (4, 'Yes'): 13, (5, 'Yes'): 1, (6, 'Yes'): 2,
        (7, 'Yes'): 3, (8, 'Yes'): 11, (12, 'Yes'): 4,
        (3, 'No'): 5, (4, 'No'): 12, (5, 'No'): 6, (6, 'No'): 7,
        (7, 'No'): 8, (8, 'No'): 10, (12, 'No'): 9,
    }