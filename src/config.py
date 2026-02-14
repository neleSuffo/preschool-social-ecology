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
    # -- General Analysis Settings --
    EXCLUSION_SECONDS = 30           # Seconds to exclude at start and end of videos
    INTERACTION_CLASSES = ['Interacting', 'Available', 'Alone']
    STATE_MAP = {"Interacting": 1, "Available": 2, "Alone": 3}
    SAMPLE_RATE = 10                 # Processing interval (only every n-th frame is analyzed)
    
    # -- Audio-Lead Interaction (Rule 1: Turn-Taking) --
    MAX_TURN_TAKING_GAP_SEC = 8      # Max gap to link KCHI and CDS into an interaction window
    MAX_SAME_SPEAKER_GAP_SEC = 1.5   # Max silence allowed between same-speaker segments before splitting
    
    # -- Visual Interaction (Rule 2: Proximity) --
    PROXIMITY_THRESHOLD = 0.8        # Min face proximity score to trigger "Interacting" via Rule 2
    INSTANT_CONFIDENCE_THRESHOLD = 0.4 # Min confidence for a single-frame detection to be "real" (Rule 2/Memory)

    # -- Sustained Audio Interaction (Rule 3: KCDS & Rule 4: KCHI + Visual) --
    SUSTAINED_KCDS_WINDOW_SEC = 1.0    # Rolling window size to check for sustained adult speech
    SUSTAINED_KCDS_THRESHOLD = 0.8  # % of KCDS required in window to activate Rule 3
    VISUAL_PERSISTENCE_SEC = 0.0     # Temporal buffer to maintain visual presence (Memory/Rule 4)
    PERSON_AUDIO_WINDOW_SEC = 2.0    # Window size for checking recent child speech in Rule 4

    # -- Presence & Hysteresis Logic (Available vs. Alone) --
    MIN_PRESENCE_CONFIDENCE_THRESHOLD = 0.28 # Entry threshold: Presence score must reach this to start "Available"
    STANDARD_EXIT_MULTIPLIER = 0.55   # Exit threshold multiplier for standard drop to "Alone" (% of entry)
    SOCIAL_COOLDOWN_EXIT_MULTIPLIER = 0.15 # Exit multiplier during social context (% of entry; bridges gaps)
    SOCIAL_CONTEXT_THRESHOLD = 0.5   # Weighted interaction history required to trigger Social Cooldown exit
    SOCIAL_COOLDOWN_SEC = 45         # Duration of the "social echo" window following an interaction
    EDGE_MARGIN = 0.05               # Normalized frame margin (%) for exit tripwire kill-switch
    
    # -- Presence Detection Gating (Robust Person Flag) --
    PERSON_AVAILABLE_WINDOW_SEC = 45 # Window size for calculating the average presence (Confidence Mass)
    MIN_PRESENCE_PERSON_FRACTION = 0.15 # Min % of detection in window for person-based availability
    MIN_PRESENCE_OHS_FRACTION = 0.03 # Min % of OHS in window for audio-based availability
    AUDIO_VISUAL_GATING_FLOOR = 0.18 # Min visual presence score required to validate OHS as "Available"
    MAX_OHS_FOR_AVAILABLE = 0.65      # % OHS above which audio is treated as noise/audiobook rather than presence

    # -- Alone Signal Robustness --
    ROBUST_ALONE_WINDOW_SEC = 2      # Window to check for consistent lack of social signals
    MAX_ALONE_FALSE_POSITIVE_FRACTION = 0.45 # Max social signals allowed in window to maintain "Alone" status

    # -- Segment Post-Processing & Reclassification --
    #MIN_INTERACTING_SEGMENT_DURATION_SEC = 2 # Min duration for an interacting segment to be kept
    #MIN_ALONE_SEGMENT_DURATION_SEC = 30     # Min duration for an alone segment to be kept
    #MIN_AVAILABLE_SEGMENT_DURATION_SEC = 12    # Min duration for an available segment to be kept
    ROLLING_SMOOTH_WINDO_SEC = 3.0          # Window size for rolling mode smoothing to remove jitter
    GAP_STRETCH_THRESHOLD = 0.1     # Max gap to automatically bridge/fill between segments
        
    INTERACTION_PERMISSION_GATE = 1.15 # Multiplier (%) for Presence Mass required to upgrade to "Interacting"

    # -- Hyperparameter Tuning Settings --
    MAX_COMBINATIONS_TUNING = 20      # Max hyperparameter combinations to test per run
    RANDOM_SAMPLING = True            # Use random sampling instead of grid search for tuning