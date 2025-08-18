import librosa
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import re
from glob import glob
from collections import Counter, defaultdict
from constants import AudioClassification, BasePaths
from config import AudioConfig

def create_jsonl_segments_from_annotations(splits, valid_rttm_classes, window_duration, window_step, output_dir):
    """
    Create JSONL training segments directly from JSON annotations without intermediate RTTM files.
    
    Parameters:
    ----------
    splits (dict): Dictionary with train/dev/test file lists
    valid_rttm_classes (list): Valid speaker classes (KCHI, CDS, OHS, SPEECH)
    window_duration (float): Duration of each training segment
    window_step (float): Step size between windows
    output_dir (Path): Output directory for JSONL files
    
    Returns:
    -------
    tuple: (segment_files, segment_counts, unique_labels, label_counts_per_split)
    """
    audio_files_dir = Path(AudioClassification.AUDIO_FILES_DIR)
    
    if not audio_files_dir.exists():
        raise FileNotFoundError(f"Audio files directory not found: {audio_files_dir}")
    
    # Generate segment file paths
    window_duration_str = str(AudioConfig.WINDOW_DURATION).replace('.', 'p')
    window_step_str = str(AudioConfig.WINDOW_STEP).replace('.', 'p')
    
    segment_files = {
        'train': output_dir / f'train_segments_w{window_duration_str}_s{window_step_str}.jsonl',
        'val': output_dir / f'val_segments_w{window_duration_str}_s{window_step_str}.jsonl',
        'test': output_dir / f'test_segments_w{window_duration_str}_s{window_step_str}.jsonl'
    }
    
    # Map split names
    split_mapping = {'dev': 'val', 'train': 'train', 'test': 'test'}
    
    segment_counts = {}
    all_unique_labels = set()
    label_counts_per_split = {}  # Track label counts per split
    
    for original_split_name, files_in_split in splits.items():
        split_name = split_mapping.get(original_split_name, original_split_name)
        print(f"\n--- Processing {split_name.title()} Data ({len(files_in_split)} files) ---")
        
        # Initialize label counts for this split
        label_counts_per_split[split_name] = Counter()
        
        # Collect all annotations for this split with timing info
        split_annotations = []
        
        for f_info in tqdm(files_in_split, desc=f"Collecting {split_name} annotations"):
            try:
                with open(f_info["path"], "r") as file_handle:
                    data = json.load(file_handle)
                
                uri = data.get('video_name', '')
                audio_path = audio_files_dir / f"{uri}.wav"
                
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                    continue
                
                # Get audio duration
                try:
                    audio_duration = librosa.get_duration(path=audio_path)
                except Exception as e:
                    print(f"Warning: Could not get duration for {audio_path}: {e}. Skipping.")
                    continue
                
                # Process annotations for this file
                file_annotations = []
                for annotation in data.get('annotations', []):
                    event_id = annotation.get('eventId', '')
                    
                    if event_id not in AudioConfig.VALID_EVENT_IDS:
                        continue
                    
                    start_sec = annotation.get('startTime', 0)
                    end_sec = annotation.get('endTime', 0)
                    duration_sec = end_sec - start_sec
                    
                    if duration_sec <= 0:
                        continue
                    
                    # Map event IDs to speaker IDs
                    speaker_ids = []
                    if event_id in ["child_talking", "singing/humming"]:
                        speaker_ids.append("KCHI")
                    elif event_id == "other_person_talking":
                        speaker_ids.append("CDS")
                    elif event_id == "overheard_speech":
                        speaker_ids.append("OHS")
                    
                    # Add SPEECH label for all valid annotations
                    if speaker_ids:
                        speaker_ids.append("SPEECH")
                    
                    for speaker_id in speaker_ids:
                        if speaker_id in valid_rttm_classes:
                            file_annotations.append({
                                'start': start_sec,
                                'end': end_sec,
                                'speaker_id': speaker_id,
                                'audio_path': str(audio_path),
                                'audio_duration': audio_duration
                            })
                
                split_annotations.extend(file_annotations)
                
            except Exception as e:
                print(f"Error processing file {f_info['path']}: {e}")
        
        # Create sliding windows from annotations
        print(f"Creating sliding windows for {len(split_annotations)} annotations...")
        
        # Group annotations by audio file
        file_annotations = defaultdict(list)
        file_durations = {}
        
        for ann in split_annotations:
            audio_path = ann['audio_path']
            file_annotations[audio_path].append(ann)
            file_durations[audio_path] = ann['audio_duration']
        
        # Generate segments for each audio file
        segment_counter = 0
        
        with open(segment_files[split_name], 'w') as f_out:
            for audio_path, annotations in tqdm(file_annotations.items(), desc=f"Generating {split_name} segments"):
                audio_duration = file_durations[audio_path]
                
                # Find the maximum time with annotations
                max_annotation_time = max(ann['end'] for ann in annotations) if annotations else 0
                analysis_end_time = min(audio_duration, max_annotation_time)
                
                # Generate sliding windows
                current_time = 0.0
                while current_time < analysis_end_time:
                    window_start = current_time
                    window_end = min(current_time + window_duration, audio_duration, analysis_end_time)
                    
                    # Find overlapping annotations
                    active_labels = set()
                    for ann in annotations:
                        # Check if annotation overlaps with window
                        if max(window_start, ann['start']) < min(window_end, ann['end']):
                            active_labels.add(ann['speaker_id'])
                    
                    # Only create segment if it has labels and valid duration
                    if active_labels and (window_end - window_start) > 0:
                        all_unique_labels.update(active_labels)
                        
                        # Count labels for this split
                        for label in active_labels:
                            label_counts_per_split[split_name][label] += 1
                        
                        segment_data = {
                            'audio_path': audio_path,
                            'start': window_start,
                            'duration': window_duration,
                            'labels': sorted(list(active_labels))
                        }
                        f_out.write(json.dumps(segment_data) + '\n')
                        segment_counter += 1
                    
                    current_time += window_step
        
        segment_counts[split_name] = segment_counter
        print(f"✓ Generated {segment_counter:,} segments for {split_name} split")
    
    return segment_files, segment_counts, sorted(list(all_unique_labels)), label_counts_per_split


def save_data_preparation_summary(segment_files, segment_counts, unique_labels, label_counts_per_split):
    """
    Save enhanced summary of data preparation results in a .txt file (instead of JSON).
    
    Parameters:
    ----------
    segment_files (dict): 
        Paths to generated segment files
    segment_counts (dict): 
        Number of segments per split
    unique_labels (list): 
        All unique labels found
    label_counts_per_split (dict): 
        Label counts per split
    """
    # Save as .txt file instead of JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = BasePaths.LOGGING_DIR / f"split_distribution_audio_{timestamp}.txt"
    
    # Build input files information
    input_files = {
        'audio_files_dir': str(AudioClassification.AUDIO_FILES_DIR),
        'participant_info_csv': str(AudioClassification.CHILDLENS_PARTICIPANT_INFO),
        'annotations_dir': str(Path(AudioClassification.CHILDLENS_PARTICIPANT_INFO).parent)
    }
    
    # Convert label counts to regular dict and calculate percentages
    label_percentages_dict = {}
    for split_name, counts in label_counts_per_split.items():
        total_labels_in_split = sum(counts.values())
        label_percentages_dict[split_name] = {
            label: (count / total_labels_in_split * 100) if total_labels_in_split > 0 else 0
            for label, count in counts.items()
        }
    
    # Create text summary
    lines = []
    lines.append("=== Data Preparation Summary ===")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("")
    
    lines.append("Configuration:")
    lines.append(f"  Window duration: {AudioConfig.WINDOW_DURATION}")
    lines.append(f"  Window step: {AudioConfig.WINDOW_STEP}")
    lines.append(f"  Sample rate: {AudioConfig.SR}")
    lines.append(f"  N mels: {AudioConfig.N_MELS}")
    lines.append(f"  Hop length: {AudioConfig.HOP_LENGTH}")
    lines.append("")
    
    lines.append("Input Files:")
    for key, path in input_files.items():
        lines.append(f"  {key}: {path}")
    
    lines.append("")
    lines.append("Output Files:")
    for key, path in segment_files.items():
        lines.append(f"  {key}: {path}")
    
    lines.append("")
    lines.append("Segment Counts:")
    for split, count in segment_counts.items():
        lines.append(f"  {split}: {count}")
    lines.append(f"  Total: {sum(segment_counts.values())}")
    
    lines.append("")
    lines.append(f"Unique Labels ({len(unique_labels)}): {', '.join(unique_labels)}")
    lines.append("")
    
    # Label counts and percentages per split
    lines.append("Label Counts and Percentages per Split:")
    for split_name in ['train', 'val', 'test']:
        if split_name in label_counts_per_split:
            lines.append(f"  {split_name.title()}:")
            for label in unique_labels:
                count = label_counts_per_split[split_name].get(label, 0)
                percentage = round(label_percentages_dict[split_name].get(label, 0.0), 2)
                lines.append(f"    {label}: {count} ({percentage}%)")
    
    # Write to text file
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines))
    
    print("\n📊 Data Preparation Summary:")
    print(f"\n✅ Summary saved to: {summary_path}")

def create_participant_splits():
    """
    Create participant ID-based splits for train/dev/test.
    
    Returns:
    -------
    dict: Dictionary with 'train', 'dev', 'test' keys containing file lists
    """
    # Load the CSV file with ID mappings
    id_mapping_df = pd.read_csv(AudioClassification.CHILDLENS_PARTICIPANT_INFO, sep=';')
    # Create mapping from file_name to ID
    file_name_to_id = dict(zip(id_mapping_df['file_name'].astype(str), id_mapping_df['child_id'].astype(str)))

    print(f"Loaded {len(file_name_to_id)} video-to-ID mappings")

    # Step 1: Load all JSON files and collect metadata with ID grouping
    id_to_files = defaultdict(list)
    files_without_id = []

    # Get the directory containing the JSON files (same directory as the CSV)
    annotations_dir = Path(AudioClassification.CHILDLENS_PARTICIPANT_INFO).parent
    json_files = glob(f"{annotations_dir}/*.json")
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            video_name = data.get('video_name', '')
            if not video_name:
                print(f"Warning: No video_name found in {json_file}")
                continue
                
            annotations = data.get('annotations', [])
            if not annotations:
                print(f"Warning: No annotations found in {json_file}")
                continue
                
            duration = max(ann.get('endTime', 0) for ann in annotations)
            
            # Extract file_name from video_name (remove extension)
            file_name = video_name.replace('.MP4', '').replace('.mp4', '')
            participant_id = file_name_to_id.get(file_name, None)
            
            file_info = {
                "path": json_file,
                "uri": video_name,
                "file_name": file_name,
                "duration": duration,
                "participant_id": participant_id
            }
            
            if participant_id:
                id_to_files[participant_id].append(file_info)
            else:
                files_without_id.append(file_info)
                print(f"Warning: No ID found for file_name '{file_name}' from video '{video_name}'")
                
        except Exception as e:
            print(f"Skipping file {json_file} due to error: {e}")

    print(f"\nFound {len(id_to_files)} unique participant IDs")
    print(f"Files without ID: {len(files_without_id)}")

    # Step 2: Calculate total duration per ID and sort IDs by total duration
    id_durations = {}
    for participant_id, files in id_to_files.items():
        total_duration = sum(f["duration"] for f in files)
        id_durations[participant_id] = total_duration

    # Sort IDs by total duration (descending) for balanced splitting
    sorted_ids = sorted(id_durations.keys(), key=lambda x: id_durations[x], reverse=True)

    # Step 3: Split IDs into train/dev/test while maintaining ratios
    total_duration = sum(id_durations.values())
    target_train = 0.8 * total_duration
    target_dev = 0.1 * total_duration
    target_test = 0.1 * total_duration

    train_ids, dev_ids, test_ids = [], [], []
    train_duration, dev_duration, test_duration = 0, 0, 0

    for participant_id in sorted_ids:
        duration = id_durations[participant_id]
        
        # Assign to the split that needs the most duration relative to its target
        train_need = max(0, target_train - train_duration)
        dev_need = max(0, target_dev - dev_duration)
        test_need = max(0, target_test - test_duration)
        
        if train_need >= dev_need and train_need >= test_need:
            train_ids.append(participant_id)
            train_duration += duration
        elif dev_need >= test_need:
            dev_ids.append(participant_id)
            dev_duration += duration
        else:
            test_ids.append(participant_id)
            test_duration += duration

    # Flatten files by split
    train_files = [f for pid in train_ids for f in id_to_files[pid]]
    dev_files = [f for pid in dev_ids for f in id_to_files[pid]]
    test_files = [f for pid in test_ids for f in id_to_files[pid]]

    splits = {
        "train": train_files,
        "dev": dev_files,
        "test": test_files
    }

    print(f"\n📊 ID-based Split Results:")
    print(f"Train: {len(train_ids)} IDs, {len(train_files)} files, {train_duration:.1f}s ({train_duration/total_duration*100:.1f}%)")
    print(f"Dev:   {len(dev_ids)} IDs, {len(dev_files)} files, {dev_duration:.1f}s ({dev_duration/total_duration*100:.1f}%)")
    print(f"Test:  {len(test_ids)} IDs, {len(test_files)} files, {test_duration:.1f}s ({test_duration/total_duration*100:.1f}%)")

    print(f"\nTrain IDs: {train_ids}")
    print(f"Dev IDs: {dev_ids}")
    print(f"Test IDs: {test_ids}")

    return splits

def main():
    """
    Streamlined audio data preparation pipeline - direct JSONL creation.
    
    This function:
    1. Processes JSON annotations and splits by participant ID
    2. Creates JSONL training segments directly from JSON annotations  
    3. Updates constants.py with generated file paths
    """
    try:
        print("🚀 Starting Streamlined Audio Data Preparation Pipeline")
        print("=" * 60)
        
        # Define output directory
        output_dir = Path(AudioClassification.INPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Create participant-based splits
        print("👥 Creating participant-based train/dev/test splits...")
        splits = create_participant_splits()
                      
        # Step 2: Create JSONL segments from annotations
        print("\n📊 Creating JSONL segments from annotations...")
        segment_files, segment_counts, unique_labels, label_counts_per_split = create_jsonl_segments_from_annotations(
            splits,
            AudioConfig.VALID_RTTM_CLASSES,
            AudioConfig.WINDOW_DURATION,
            AudioConfig.WINDOW_STEP,
            output_dir
        )
        
        # Check if any segments were created
        total_segments = sum(segment_counts.values())
        if total_segments == 0:
            raise ValueError("No training segments were created. Check your annotations and audio files.")
        
        # Save summary
        save_data_preparation_summary(segment_files, segment_counts, unique_labels, label_counts_per_split)
        
        print(f"\n✅ Streamlined data preparation pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Data preparation pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()