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
    CUT_VIDEO = ['quantex_at_home_id255237_2022_05_08_04']
    CUT_VIDEO_OFFSET = 7123 #add offset to frame number for this video
    SHIFTED_VIDEOS_OFFSETS = {'quantex_at_home_id257511_2021_07_13_01': (24, -6), # annotations are shifted by six frames from frame 30 on
                              'quantex_at_home_id257573_2021_04_02_01': (28, -2), #annotations are shifted by two frames from frame 28 on
                              'quantex_at_home_id257578_2021_05_12_04': (28, -2), #annotations are shifted by two frames from frame 28 on
                              'quantex_at_home_id257578_2021_05_12_05': (28, -2),
                              'quantex_at_home_id257578_2021_05_12_06': (28, -2),
                              'quantex_at_home_id257578_2021_05_16_01': (28, -2)} #annotations are shifted by two frames from frame 28 on
    NON_STANDARD_FRAME_STEPS = {'quantex_at_home_id258704_2022_05_15_01': 34} # Interval is 34 frames, not 30
    

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

class BookConfig:
    """Configuration for book detection."""
    MODEL_NAME = "yolo12l"
    DATABASE_CATEGORY_IDS = [73]  # COCO category ID for 'book'
    MODEL_ID = 5
    
# Specific Task Configurations
class PersonConfig:
    """Configuration for person detection and classification."""
    MODEL_SIZE = 'l'  # Default model size
    MODEL_NAME = f"yolo12{MODEL_SIZE}"
    # Ratio of training data to use for training
    TRAIN_SPLIT_RATIO = 0.6
    # Ratio of class-to-class samples in each dataset split
    MAX_CLASS_RATIO_THRESHOLD = 0.9
    AGE_GROUP_TO_CLASS_ID_PERSON_ONLY = {
        'Inf': 0,
        'Child': 0,
        'Teen': 0,
        'Adult': 0,
        'Dk': 0,
    }
    AGE_GROUP_TO_CLASS_ID_AGE_BINARY = {
        'Inf': 0,
        'Child': 0,
        'Teen': 1,
        'Adult': 1,
    }
    MODEL_CLASS_ID_TO_LABEL_AGE_BINARY = {
        0: "child",
        1: "adult"
    }
    MODEL_CLASS_ID_TO_LABEL_PERSON_ONLY = {0: "person"}
    DATABASE_CATEGORY_IDS = [1, 2]
    TARGET_LABELS_AGE_BINARY = ['child', 'adult']
    TARGET_LABELS_PERSON_ONLY = ['person']
    TARGET_LABELS_CLS = ['0_no_person', '1_person']
    NEGATIVE_SAMPLING_RATIO = 1 #100% negative samples compared to positive samples in train and val splits

    MIN_IDS_PER_SPLIT = 2

    NUM_EPOCHS = 300
    BATCH_SIZE = 20
    IMG_SIZE = 832
    
    # number of videos per batch
    LR = 1e-3
    FREEZE_CNN = True
    PATIENCE = 15
    SEQUENCE_LENGTH = 60
    DROPOUT = 0.5
    WEIGHT_DECAY = 1e-5
    FEAT_DIM = 512
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    BIDIRECTIONAL = True
    NUM_OUTPUTS = 1
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
    AGE_GROUP_TO_CLASS_ID_AGE_BINARY = {
        'infant': 0,
        'child': 0,
        'teen': 1,
        'adult': 1,
    }
    AGE_GROUP_TO_CLASS_ID_FACE_ONLY = {
        'infant': 0,
        'child': 0,
        'teen': 0,
        'adult': 0,
    }
    MODEL_CLASS_ID_TO_LABEL_AGE_BINARY = {
        0: "child",
        1: "adult"
    }
    MODEL_CLASS_ID_TO_LABEL_FACE_ONLY = {0: "face"}
    DATABASE_CATEGORY_IDS = [10]
    TARGET_LABELS_AGE_BINARY = ['child', 'adult']
    TARGET_LABELS_FACE_ONLY = ['face']
    NEGATIVE_SAMPLING_RATIO = 1 #100% negative samples compared to positive samples in train and val splits

    TRAIN_SPLIT_RATIO = 0.6
    MIN_IDS_PER_SPLIT = 2
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
    VALID_RTTM_CLASSES = ['OHS', 'KCDS', 'KCHI']
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
    EXCLUSION_SECONDS = 30 # seconds to exclude at start and end of videos
    INTERACTION_CLASSES = ['Interacting', 'Available', 'Alone']
    SAMPLE_RATE = 10 # every n-th frame is processed
    
    # -- Face Related Parameters --
    PROXIMITY_THRESHOLD = 0.78 # face proximity so that frame is counted as interaction
    
    # -- Person Related Parameters --
    PERSON_AVAILABLE_WINDOW_SEC = 10 # window duration for is_sustained_person_or_face_present (rule available)
    MIN_PRESENCE_PERSON_FRACTION = 0.05 # # At least 5% presence in PERSON_AVAILABLE_WINDOW_SEC window
    MIN_PRESENCE_OHS_FRACTION = 0.025 # # At least 5% presence in PERSON_AVAILABLE_WINDOW_SEC window
    PERSON_AUDIO_WINDOW_SEC = 2 # window duration for rule4_person_recent_speech
    KCHI_PERSON_BUFFER_FRAMES = 10 # number of frames to look back and forward for KCHI + visual presence

    # -- Audio Related Parameters --
    MAX_SAME_SPEAKER_GAP_SEC = 2.25 # maximum gap duration to consider same speaker segment
    MIN_KCDS_DURATION_SEC = 2 # minimum duration of KCDS to consider for analysis
    MAX_TURN_TAKING_GAP_SEC = 5 # maximum gap duration for turn-taking analysis
    SUSTAINED_KCDS_SEC = 3 # consecutive seconds of KCDS to activate rule3_kcds_speaking 


    # --- Media Related Parameters --
    MEDIA_WINDOW_SEC = 15  # Time window for sustained 'Media' check
    MIN_BOOK_PRESENCE_FRACTION = 0.7  # At least 70% media presence in the MEDIA_WINDOW_SEC window
    MIN_PRESENCE_OHS_KCDS_FRACTION_MEDIA = 0.05  # At least 5% OHS/KCDS presence in the MEDIA_WINDOW_SEC window
    MAX_KCHI_FRACTION_FOR_MEDIA = 0.1  # Maximum fraction of KCHI presence allowed for media interaction
    
    
    # -- Segment Merging Parameters --
    MIN_INTERACTING_SEGMENT_DURATION_SEC = 0.7 # minimum duration for a interacting segment
    MIN_ALONE_SEGMENT_DURATION_SEC = 5 # minimum duration for an alone segment
    MIN_AVAILABLE_SEGMENT_DURATION_SEC = 4 # minimum duration for an available segment
    MIN_ALONE_SANDWICH_DURATION_SEC = 4 # minimum duration for alone segments sandwiched between interacting segments
    MIN_INTERACTING_SANDWICH_DURATION_SEC = 4 # minimum duration for interacting segments sandwiched between alone segments
    MIN_RECLASSIFY_DURATION_SEC = 5 # minimum duration for reclassifying 'Available' segments
    ALONE_RECLASSIFY_VISUAL_THRESHOLD = 0.24 # Percentage of visual presence in segments available for reclassification to alone
    ALONE_RECLASSIFY_AUDIO_THRESHOLD = 0.24 # Percentage of audio presence in segments available for reclassification to alone
    KCHI_ONLY_FRACTION_THRESHOLD = 0.75 # Percentage of KCHI-only frames in segments available or alone for reclassification
    MIN_PERSON_PRESENCE_FRACTION = 0.04 # At least 4% person presence in segments available or alone for reclassification
    
    ROBUST_ALONE_WINDOW_SEC = 10  # Time window for sustained 'Alone' check (e.g., 7.5 seconds)
    MAX_ALONE_FALSE_POSITIVE_FRACTION = 0.05  # Max fraction (5%) of social signal frames allowed in the window for classification as 'Alone'
    # -- Hyperparameter Tuning Parameters --
    MAX_COMBINATIONS_TUNING = 20 # Maximum number of hyperparameter combinations to tune
    RANDOM_SAMPLING = True # Whether to use random sampling for hyperparameter tuning