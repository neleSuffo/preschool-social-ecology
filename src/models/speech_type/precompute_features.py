import json
import numpy as np
from pathlib import Path
from config import AudioConfig
from constants import AudioClassification
from utils import extract_features

def process_segments_file(segments_file, output_subfolder):
    parent_dir = Path(segments_file).parent
    cache_dir = Path(parent_dir / "feature_cache" / output_subfolder)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Calculate fixed_time_steps exactly as in training
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))

    with open(segments_file, 'r') as f:
        for line in f:
            segment = json.loads(line.strip())
            audio_path = segment['audio_path']
            start = segment['start']
            duration = segment['duration']
            segment_id = segment.get('id') or f"{Path(audio_path).stem}_{start}_{duration}"

            # Use config parameters for extraction
            feature = extract_features(
                audio_path, start, duration,
                sr=AudioConfig.SR,
                n_mels=AudioConfig.N_MELS,
                hop_length=AudioConfig.HOP_LENGTH,
                fixed_time_steps=fixed_time_steps
            )

            cache_path = cache_dir / f"{segment_id}.npy"
            np.save(cache_path, feature)

def main():
    # Segment files from constants
    train_segments_file = AudioClassification.TRAIN_SEGMENTS_FILE
    val_segments_file = AudioClassification.VAL_SEGMENTS_FILE
    test_segments_file = AudioClassification.TEST_SEGMENTS_FILE

    print("\n📂 Processing TRAIN segments...")
    process_segments_file(train_segments_file, "train")

    print("\n📂 Processing VALIDATION segments...")
    process_segments_file(val_segments_file, "val")

    print("\n📂 Processing TEST segments...")
    process_segments_file(test_segments_file, "test")

    print("\n🎉 All segments processed and features saved successfully.")

if __name__ == "__main__":
    main()