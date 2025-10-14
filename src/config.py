from collections import defaultdict
from dynaconf import Dynaconf
from pathlib import Path
import numpy as np

# Dynaconf settings
SETTINGS = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

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
    BATCH_SIZE = 4
    LR = 1e-3
    FREEZE_CNN = True
    PATIENCE = 15
    SEQUENCE_LENGTH = 60
    DROPOUT = 0.5
    WEIGHT_DECAY = 1e-5
    FEAT_DIM = 512
    RNN_HIDDEN = 512
    RNN_LAYERS = 3
    BIDIRECTIONAL = True
    NUM_OUTPUTS = 2
    BACKBONE = 'efficientnet_b0'
    CONFIDENCE_THRESHOLD = 0.5
    BATCH_SIZE_INFERENCE = 64
    WINDOW_SIZE = 6 # frames (6 extracted frames = 60 original frames = 2 seconds)
    STRIDE = 1 # frames (every 1st extracted frame = every 10 original frames)
    CHILD_POS_WEIGHT = 3.08  
    ADULT_POS_WEIGHT = 2.60
    MODEL_ID = 2

class FaceConfig:
    """Configuration for face detection and classification."""
    MODEL_SIZE = 'l'  # Default model size
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
    
    CLUSTER_CONSECUTIVE_FRAMES = 1
    REPRESENTATIVE_BLUR_THRESHOLD = 60
    MIN_BLUR_THRESHOLD = 20
    DEEPFACE_BACKEND = "retinaface"
    FINAL_CONFIRMATION_DISTANCE_THRESHOLD = 0.6
    VERIFIED_DISTANCE_THRESHOLD = 0.68

class AudioConfig:
    """Configuration for audio classification."""
    VALID_RTTM_CLASSES = ['OHS', 'CDS', 'KCHI']
    VALID_EVENT_IDS = {"child_talking", "other_person_talking", "overheard_speech", "singing/humming"}
    SR = 16000
    N_MELS = 256
    HOP_LENGTH = 256
    #WINDOW_DURATION = 3.0
    #WINDOW_STEP = 1.0
    MAX_SEGMENT_DURATION = 20 #in seconds
    FIXED_TIME_STEPS = int(np.ceil(MAX_SEGMENT_DURATION * SR / HOP_LENGTH))

    NUM_EPOCHS = 200
    PATIENCE = 30
    MODEL_ID = 3

class KchiVoc_Config:
    """Configuration for KCHI vocalizations."""
    MODEL_ID = 4
    
class InferenceConfig:
    """Configuration for inference settings."""
    INTERACTION_CLASSES = ['Interacting', 'Co-present', 'Alone']
    SAMPLE_RATE = 10 # every n-th frame is processed
    PROXIMITY_THRESHOLD = 0.7 # face proximity so that frame is counted as interaction
    MIN_SEGMENT_DURATION_SEC = 5 # minimum duration for a segment to be considered
    MIN_CHANGE_DURATION_SEC = 3 # minimum duration for a change to be considered
    SPEECH_CLASSES = ['KCHI', 'FEM_MAL']
    TURN_TAKING_BASE_WINDOW_SEC = 10 # base window duration for turn-taking analysis
    TURN_TAKING_EXT_WINDOW_SEC = 15 # extended window duration for turn-taking analysis
    MAX_TURN_TAKING_GAP_SEC = 5 # maximum gap duration for turn-taking analysis
    PERSON_AUDIO_WINDOW_SEC = 10 # window duration for person audio analysis
    GAP_MERGE_DURATION_SEC = 5 # duration for merging gaps in interaction segments
    VALIDATION_SEGMENT_DURATION_SEC = 10 # min duration for validation segments
    PERSON_PRESENT_THRESHOLD = 0.05 # threshold for considering a person present in a window segment with only audio turn taking
    EVALUATION_IOU = 0.5 # IoU threshold for evaluation
    MAX_COMBINATIONS_TUNING = 20 # Maximum number of hyperparameter combinations to tune
    RANDOM_SAMPLING = True # Whether to use random sampling for hyperparameter tuning
