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

# Specific Task Configurations
class PersonConfig:
    """Configuration for person detection and classification."""
    YOLO_DETECTION_TARGET = "person"
    PERSON_CLS_MAPPING = {
        0: "adult_person",
        1: "child_person"
    }
    PERSON_CLS = {
        'Inf': 0,
        'child': 0,
        'teen': 1,
        'adult': 1,
    }
    PERSON_FACE_DET = {
        0: 0,
        1: 0,
        10: 1,
        11: 2
    }
    PERSON_CLS_TARGET_CLASS_IDS = [1, 2]

class FaceConfig:
    """Configuration for face detection and classification."""
    FACE_CLS = {
        'infant': 0,
        'child': 0,
        'teen': 1,
        'adult': 1,
    }
    FACE_CLS_TARGET_CLASS_IDS = [10]

class AudioConfig:
    """Configuration for audio classification."""
    VALID_RTTM_CLASSES = ['OHS', 'CDS', 'KCHI', 'SPEECH']

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