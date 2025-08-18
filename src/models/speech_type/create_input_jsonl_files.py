# filepath: /home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/models/speech_type/prepare_audio_data.py
import librosa
import numpy as np
import pandas as pd
import json
import datetime
from tqdm import tqdm
from pathlib import Path
import re
from constants import AudioClassification, DataPaths, BasePaths
from config import AudioConfig

def convert_rttm_to_training_segments(rttm_path, audio_files_dir, valid_rttm_classes, window_duration, window_step, sr, n_mels, hop_length, output_segments_path):
    """
    Parse RTTM annotation files into windowed multi-label training segments.
    
    Converts continuous speaker/activity annotations into fixed-duration overlapping
    windows suitable for supervised multi-label audio classification training.
    
    Parameters:
    ----------
    rttm_path (str):
        Path to RTTM annotation file with speaker/activity labels
    audio_files_dir (str):
        Directory containing corresponding audio files (.wav)
    valid_rttm_classes (list):
        List of valid speaker/activity IDs to include (e.g., ['OHS', 'CDS', 'KCHI'])
    window_duration (float):
        Duration of each training segment in seconds (e.g., 3.0)
    window_step (float):
        Step size between windows in seconds (e.g., 1.0 for overlap)
    sr (int):
        Sample rate for audio processing (used for validation)
    n_mels (int):
        Number of mel filters (used for validation)
    hop_length (int):
        Hop length for feature extraction (used for validation)
    output_segments_path (str):
        Path where segment metadata will be saved (JSONL format)
    
    Returns:
    -------
    tuple: (segment_count, unique_labels_found)
        segment_count (int): Number of valid segments created
        unique_labels_found (list): Sorted list of unique labels found in data
        
    Output format (JSONL):
        Each line contains: {"audio_path": str, "start": float, "duration": float, "labels": list}
    
    Processing logic:
        1. Load RTTM file with speaker/activity time annotations
        2. For each audio file, create sliding windows of specified duration
        3. For each window, find all overlapping speaker/activity labels
        4. Keep only windows with at least one valid label
        5. Save segment metadata for training data generators
        
    RTTM format expected:
        SPEAKER file_id channel start duration NA1 NA2 speaker_id NA3 NA4
        
    Note:
        Handles audio duration mismatches and missing files gracefully.
        Progress bar shows processing status across all audio files.
    """
    try:
        rttm_df = pd.read_csv(rttm_path, sep=' ', header=None, 
                              names=['type', 'file_id', 'channel', 'start', 'duration', 'NA1', 'NA2', 'speaker_id', 'NA3', 'NA4'])
    except Exception as e:
        print(f"Error reading RTTM file {rttm_path}: {e}")
        return [], []
    
    all_unique_labels = set()
    unique_file_ids = rttm_df['file_id'].unique()
    print(f"Processing {len(unique_file_ids)} audio files from RTTM: {Path(rttm_path).name}...")
    
    segment_counter = 0
    with open(output_segments_path, 'w', newline='') as f_out:
        for file_id in tqdm(unique_file_ids):
            audio_path = Path(audio_files_dir) / f"{file_id}.wav"
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                continue
            
            file_segments = rttm_df[rttm_df['file_id'] == file_id].copy()
            file_segments['end'] = file_segments['start'] + file_segments['duration']
            
            try:
                audio_duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_path}: {e}. Skipping.")
                continue
                
            max_time_rttm = file_segments['end'].max() if not file_segments.empty else 0
            analysis_end_time = min(audio_duration, max_time_rttm)
            
            current_time = 0.0
            while current_time < analysis_end_time:
                window_start = current_time
                window_end = min(current_time + window_duration, audio_duration, analysis_end_time)
                
                active_speaker_ids = set()
                for _, row in file_segments.iterrows():
                    segment_start = row['start']
                    segment_end = row['end']
                    if max(window_start, segment_start) < min(window_end, segment_end):
                        active_speaker_ids.add(row['speaker_id'])
                
                active_labels = {sid for sid in active_speaker_ids if sid in valid_rttm_classes}
                all_unique_labels.update(active_labels)
                
                if active_labels and (window_end - window_start) > 0:
                    segment_data = {
                        'audio_path': str(audio_path),
                        'start': window_start,
                        'duration': window_duration,
                        'labels': sorted(list(active_labels))
                    }
                    f_out.write(json.dumps(segment_data) + '\n')
                    segment_counter += 1
                
                current_time += window_step
                
    return segment_counter, sorted(list(all_unique_labels))

def process_rttm_data_splits(output_dir: Path):
    """
    Process RTTM files for all data splits (train/val/test) with ID-based splitting already applied.
    
    Parameters:
    ----------
    output_dir (Path): Directory to save processed segment files
        
    Returns:
    -------
    tuple: (segment_files, segment_counts, unique_labels) where:
           - segment_files: dict with paths to generated segment files
           - segment_counts: dict with counts per split
           - unique_labels: set of all unique labels found across splits
    """
    # RTTM file paths (already ID-split)
    rttm_files = {
        'train': AudioClassification.TRAIN_RTTM_FILE,
        'val': AudioClassification.VAL_RTTM_FILE,
        'test': AudioClassification.TEST_RTTM_FILE
    }
    
    audio_files_dir = Path(AudioClassification.AUDIO_FILES_DIR)
    
    if not audio_files_dir.exists():
        raise FileNotFoundError(f"Audio files directory not found: {audio_files_dir}")
    
    wav_files = list(audio_files_dir.glob("*.wav"))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in directory: {audio_files_dir}")
        
    for split_name, rttm_path in rttm_files.items():
        if not rttm_path.exists():
            raise FileNotFoundError(f"RTTM file not found for {split_name} split: {rttm_path}")

    # Generate segment file paths
    window_duration_str = str(AudioConfig.WINDOW_DURATION).replace('.', 'p')
    window_step_str = str(AudioConfig.WINDOW_STEP).replace('.', 'p')
    
    segment_files = {
        'train': output_dir / f'train_segments_w{window_duration_str}_s{window_step_str}.jsonl',
        'val': output_dir / f'val_segments_w{window_duration_str}_s{window_step_str}.jsonl',
        'test': output_dir / f'test_segments_w{window_duration_str}_s{window_step_str}.jsonl'
    }
    
    segment_counts = {}
    all_unique_labels = set()
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n--- Processing {split_name.title()} Data ---")
        
        num_segments, unique_labels = convert_rttm_to_training_segments(
            rttm_files[split_name], 
            audio_files_dir, 
            AudioConfig.VALID_RTTM_CLASSES,
            AudioConfig.WINDOW_DURATION, 
            AudioConfig.WINDOW_STEP, 
            AudioConfig.SR, 
            AudioConfig.N_MELS, 
            AudioConfig.HOP_LENGTH, 
            segment_files[split_name]
        )
        
        segment_counts[split_name] = num_segments
        all_unique_labels.update(unique_labels)
        
        if num_segments == 0:
            print(f"Warning: No segments found for {split_name} split. Check RTTM file: {rttm_files[split_name]}")
        else:
            print(f"✓ Generated {num_segments:,} segments for {split_name} split")
    
    return segment_files, segment_counts, sorted(list(all_unique_labels))

def save_data_preparation_summary(output_dir, segment_files, segment_counts, unique_labels):
    """
    Save summary of data preparation results.
    
    Parameters:
    ----------
    output_dir (Path): Output directory for saving summary
    segment_files (dict): Paths to generated segment files
    segment_counts (dict): Number of segments per split
    unique_labels (list): All unique labels found
    """
    output_dir = Path(output_dir)
    
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'configuration': {
            'window_duration': AudioConfig.WINDOW_DURATION,
            'window_step': AudioConfig.WINDOW_STEP,
            'sample_rate': AudioConfig.SR,
            'n_mels': AudioConfig.N_MELS,
            'hop_length': AudioConfig.HOP_LENGTH,
            'valid_rttm_classes': AudioConfig.VALID_RTTM_CLASSES
        },
        'input_files': {
            'audio_files_dir': str(AudioClassification.AUDIO_FILES_DIR),
            'train_rttm': str(AudioClassification.TRAIN_RTTM_FILE),
            'val_rttm': str(AudioClassification.VAL_RTTM_FILE),
            'test_rttm': str(AudioClassification.TEST_RTTM_FILE)
        },
        'output_files': {k: str(v) for k, v in segment_files.items()},
        'segment_counts': segment_counts,
        'unique_labels': unique_labels,
        'total_segments': sum(segment_counts.values())
    }
    
    summary_path = BasePaths.LOGGING_DIR / 'audio_data_preparation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n📊 Data Preparation Summary:")
    print(f"  Training segments: {segment_counts.get('train', 0):,}")
    print(f"  Validation segments: {segment_counts.get('val', 0):,}")
    print(f"  Test segments: {segment_counts.get('test', 0):,}")
    print(f"  Total segments: {sum(segment_counts.values()):,}")
    print(f"  Unique labels: {len(unique_labels)} {unique_labels}")
    print(f"\n✅ Summary saved to: {summary_path}")

def update_constants_with_segment_paths(segment_files):
    """
    Update the constants.py file with the generated segment file paths.
    
    This function writes the generated segment file paths back to the constants.py
    file, overwriting existing values for TRAIN_SEGMENTS_FILE, VAL_SEGMENTS_FILE,
    and TEST_SEGMENTS_FILE in the AudioClassification class.
    
    Parameters:
    ----------
    segment_files (dict): Dictionary containing paths to generated segment files
                         with keys 'train', 'val', 'test'
    """   
    # Find the constants.py file
    constants_file = Path(__file__).parent.parent.parent / 'constants.py'
    
    if not constants_file.exists():
        print(f"Warning: constants.py not found at {constants_file}")
        return False
    
    try:
        # Read the current constants file
        with open(constants_file, 'r') as f:
            content = f.read()
        
        # Convert paths to relative paths using INPUT_DIR
        train_filename = Path(segment_files['train']).name
        val_filename = Path(segment_files['val']).name
        test_filename = Path(segment_files['test']).name
        
        train_path_str = f'Path(INPUT_DIR / "{train_filename}")'
        val_path_str = f'Path(INPUT_DIR / "{val_filename}")'
        test_path_str = f'Path(INPUT_DIR / "{test_filename}")'
        
        # Update or add the segment file paths in AudioClassification class
        # Pattern to find the AudioClassification class
        class_pattern = r'(class AudioClassification[^:]*:.*?)(?=class|\Z)'
        class_match = re.search(class_pattern, content, re.DOTALL)
        
        if class_match:
            class_content = class_match.group(1)
            
            # Update existing or add new segment file variables
            patterns_and_replacements = [
                (r'(\s+)TRAIN_SEGMENTS_FILE\s*=.*', rf'\1TRAIN_SEGMENTS_FILE = {train_path_str}'),
                (r'(\s+)VAL_SEGMENTS_FILE\s*=.*', rf'\1VAL_SEGMENTS_FILE = {val_path_str}'),
                (r'(\s+)TEST_SEGMENTS_FILE\s*=.*', rf'\1TEST_SEGMENTS_FILE = {test_path_str}')
            ]
            
            updated_class_content = class_content
            variables_added = []
            
            for pattern, replacement in patterns_and_replacements:
                if re.search(pattern, updated_class_content):
                    updated_class_content = re.sub(pattern, replacement, updated_class_content)
                    var_name = replacement.split('=')[0].strip()
                    variables_added.append(var_name)
                else:
                    # Add the variable at the end of the class if it doesn't exist
                    var_name = replacement.split('=')[0].strip()
                    var_line = f"    {replacement.split('=')[0].strip()} = {replacement.split('=')[1].strip()}\n"
                    
                    # Find the last line of the class (before next class or end of file)
                    lines = updated_class_content.split('\n')
                    insert_pos = len(lines) - 1
                    
                    # Find the last non-empty line that's indented (part of the class)
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip() and (lines[i].startswith('    ') or lines[i].startswith('\t')):
                            insert_pos = i + 1
                            break
                    
                    lines.insert(insert_pos, var_line.rstrip())
                    updated_class_content = '\n'.join(lines)
                    variables_added.append(f"Added {var_name}")
            
            # Replace the class content in the full file content
            updated_content = content.replace(class_content, updated_class_content)
            
            # Write back to the file
            with open(constants_file, 'w') as f:
                f.write(updated_content)
            
            print(f"\n🔄 Updated constants.py:")
            for var_change in variables_added:
                print(f"  ✓ {var_change}")
            print(f"  📁 File: {constants_file}")
            
            return True
            
        else:
            print("Warning: AudioClassification class not found in constants.py")
            return False
            
    except Exception as e:
        print(f"Error updating constants.py: {e}")
        return False

def main():
    """
    Main function for audio data preparation.
    """   
    try:
        print("🚀 Starting Audio Data Preparation Pipeline")
        print("=" * 60)
        
        output_dir = Path(AudioClassification.INPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data preparation output directory: {output_dir}")
        
        # Process RTTM data for all splits
        print("📊 Processing RTTM data splits...")
        segment_files, segment_counts, unique_labels = process_rttm_data_splits(output_dir)
        
        # Update constants.py with the generated segment file paths
        print("🔄 Updating constants.py with segment file paths...")
        update_success = update_constants_with_segment_paths(segment_files)
        
        # Save summary
        save_data_preparation_summary(output_dir, segment_files, segment_counts, unique_labels)
        
        print(f"\n✅ Data preparation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"\n🔗 To use these prepared files for training, run:")
        print(f"   python train_audio_classifier_main.py --data_dir {output_dir}")
        
    except Exception as e:
        print(f"❌ Data preparation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()