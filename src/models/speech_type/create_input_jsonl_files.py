"""
Audio Data Preparation Pipeline for Multi-Label Voice Type Classification

This module creates training data segments in JSONL format directly from JSON annotations,
implementing participant ID-based data splitting to prevent data leakage. The pipeline
processes ChildLens longitudinal study annotations and creates sliding window segments
suitable for CNN-RNN hybrid model training.

Key Features:
- Direct JSON-to-JSONL conversion bypassing intermediate RTTM files
- Participant ID-based train/validation/test splits (80/10/10)
- Sliding window segmentation with configurable duration and overlap
- Multi-label annotation support (KCHI, CDS, OHS)
- Comprehensive data distribution reporting and validation
- Duration-balanced splitting to ensure representative data across splits

Event ID Mapping:
- child_talking, singing/humming → KCHI (Key Child)
- other_person_talking → CDS (Child-Directed Speech)  
- overheard_speech → OHS (Overheard Speech)

Dependencies:
- librosa: Audio duration extraction and validation
- pandas: CSV participant mapping processing
- json: Annotation file parsing and JSONL output
- tqdm: Progress tracking for batch operations
- pathlib: Cross-platform path handling
- collections: Efficient counting and grouping operations
"""

import librosa
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from glob import glob
from collections import Counter, defaultdict
from constants import AudioClassification, BasePaths
from config import AudioConfig

def map_event_to_speaker_ids(event_id):
    """
    Map ChildLens annotation event IDs to standardized speaker classifications.
    
    This mapping transforms longitudinal study event categories into 
    voice type labels suitable for multi-label classification training.
    
    Event Categories:
    - child_talking, singing/humming: Child vocalizations (KCHI)
    - other_person_talking: Speech directed at child (CDS)
    - overheard_speech: Adult-adult conversations (OHS)
    
    Parameters:
    ----------
    event_id (str): ChildLens annotation event identifier
        
    Returns:
    -------
    list: Speaker IDs corresponding to the event type
    """
    speaker_ids = []
    
    # Map specific event types to voice categories
    if event_id in ["child_talking", "singing/humming"]:
        speaker_ids.append("KCHI")  # Key Child vocalizations
    elif event_id == "other_person_talking":
        speaker_ids.append("CDS")   # Child-Directed Speech  
    elif event_id == "overheard_speech":
        speaker_ids.append("OHS")   # Overheard Speech
        
    return speaker_ids

def extract_file_name_from_video_name(video_name):
    """
    Extract base filename from video name by removing file extensions.
    
    Handles both uppercase and lowercase MP4 extensions commonly
    found in longitudinal video datasets.
    
    Parameters:
    ----------
    video_name (str): Video filename with extension
        
    Returns:
    -------
    str: Base filename without extension
    """
    return video_name.replace('.MP4', '').replace('.mp4', '')

def create_jsonl_segments_from_annotations(splits, valid_rttm_classes, output_dir):
    """
    Create JSONL training segments directly from JSON annotations using Ground Truth (GT) events as segments.
    Each segment corresponds to the temporal boundaries of an original KCHI, CDS, or OHS event. 
    This approach preserves the natural event durations and supports multi-label assignments
    when events overlap.
    
    Segment Strategy:
    - Segment boundaries (start/end) are defined by the start/end time of a GT event.
    - Segment duration is the actual duration of the event (variable length).
    - Multi-label support is retained by checking for overlapping GT events.
    
    Parameters:
    ----------
    splits (dict): 
        Participant-based data splits {'train': files, 'dev': files, 'test': files}
    valid_rttm_classes (list): 
        Valid speaker classifications for filtering (KCHI, CDS, OHS)
    output_dir (Path): 
        Directory to save the output JSONL files

    Returns:
    -------
    tuple: 
        (segment_files, segment_counts, unique_labels, label_counts_per_split)
        - segment_files: Dict mapping split names to JSONL file paths
        - segment_counts: Number of segments generated per split  
        - unique_labels: All voice type labels found in data
        - label_counts_per_split: Label frequency statistics per split
    """
    # Validate audio files directory exists
    audio_files_dir = Path(AudioClassification.CHILDLENS_AUDIO_DIR)
    if not audio_files_dir.exists():
        raise FileNotFoundError(f"Audio files directory not found: {audio_files_dir}")
    
    # Generate simplified segment file paths, reflecting the new segment type
    segment_files = {
        'train': output_dir / f'train_segments.jsonl',
        'val': output_dir / f'val_segments.jsonl',
        'test': output_dir / f'test_segments.jsonl'
    }
    
    # Initialize tracking variables for data distribution analysis
    split_mapping = {'dev': 'val', 'train': 'train', 'test': 'test'}
    segment_counts = {}
    all_unique_labels = set()
    label_counts_per_split = {}
    
    # Process each data split (train/validation/test) separately
    for original_split_name, files_in_split in splits.items():
        split_name = split_mapping.get(original_split_name, original_split_name)
        print(f"\n--- Processing {split_name.title()} Data ({len(files_in_split)} files) ---")
        
        # Initialize label frequency tracking for distribution analysis
        label_counts_per_split[split_name] = Counter()
        
        # Group annotations by audio file for efficient processing
        file_annotations_by_audio_path = defaultdict(list)
        file_durations = {}
        
        # Process each annotation file in the current split to collect all GT events
        for f_info in tqdm(files_in_split, desc=f"Collecting {split_name} GT events"):
            try:
                # Load JSON annotation data
                with open(f_info["path"], "r") as file_handle:
                    data = json.load(file_handle)
                
                # Extract video name and construct corresponding audio file path
                uri = data.get('video_name', '')
                audio_path = audio_files_dir / f"{uri}.wav"
                
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                    continue
                
                try:
                    audio_duration = librosa.get_duration(path=audio_path)
                    file_durations[str(audio_path)] = audio_duration
                except Exception as e:
                    print(f"Warning: Could not get duration for {audio_path}: {e}. Skipping.")
                    continue
                
                # Parse annotations and convert to standardized speaker classifications
                for annotation in data.get('annotations', []):
                    event_id = annotation.get('eventId', '')
                    # Filter to only valid event types defined in configuration
                    if event_id not in AudioConfig.VALID_EVENT_IDS:
                        continue
                    
                    start_sec = annotation.get('startTime', 0)
                    end_sec = annotation.get('endTime', 0)
                    duration_sec = end_sec - start_sec
                    
                    if duration_sec <= 0:
                        continue
                    
                    speaker_ids = map_event_to_speaker_ids(event_id)
                    
                    for speaker_id in speaker_ids:
                        if speaker_id in valid_rttm_classes:
                            # Store the raw GT event for later processing
                            file_annotations_by_audio_path[str(audio_path)].append({
                                'start': start_sec,
                                'end': end_sec,
                                'speaker_id': speaker_id
                            })
                            
            except Exception as e:
                print(f"Error processing file {f_info['path']}: {e}")
        
        # Generate segments using the GT event boundaries
        print(f"Creating segments based on {len(file_annotations_by_audio_path)} files with GT events...")
        segment_counter = 0
        
        # Write segments directly to JSONL file for memory efficiency
        with open(segment_files[split_name], 'w') as f_out:
            # Iterate over each audio file that has annotations
            for audio_path, annotations in tqdm(file_annotations_by_audio_path.items(), desc=f"Generating {split_name} segments"):
                # Use a set to track unique segments generated (start/end/labels) 
                # to prevent duplicating the same multi-label segment if events perfectly overlap.
                unique_segments = set() 
                
                for primary_ann in annotations:
                    window_start = primary_ann['start']
                    window_end = primary_ann['end']
                    segment_duration = window_end - window_start
                    
                    if segment_duration <= 0.01:
                        continue
                        
                    active_labels = set()
                    
                    # Check for ALL overlapping annotations (including the primary one)
                    # This ensures multi-label assignment if events overlap
                    for secondary_ann in annotations:
                        # Check temporal overlap between secondary annotation and primary segment
                        # Uses: max(start_times) < min(end_times)
                        if max(window_start, secondary_ann['start']) < min(window_end, secondary_ann['end']):
                            active_labels.add(secondary_ann['speaker_id'])

                    sorted_labels = tuple(sorted(list(active_labels)))
                    segment_key = (window_start, window_end, sorted_labels)
                    
                    # Only process this unique segment once
                    if segment_key in unique_segments:
                        continue
                    unique_segments.add(segment_key)
                    
                    # Update global label tracking for dataset statistics
                    all_unique_labels.update(active_labels)
                    
                    # Update per-split label frequency counts for distribution analysis
                    for label in active_labels:
                        label_counts_per_split[split_name][label] += 1
                    
                    # Create segment data structure for JSONL output
                    segment_data = {
                        'audio_path': audio_path,
                        'start': window_start,
                        'duration': segment_duration,  # The actual, variable duration of the event
                        'labels': sorted(list(active_labels))
                    }
                    f_out.write(json.dumps(segment_data) + '\n')
                    segment_counter += 1

        # Record segment count for this split and display progress
        segment_counts[split_name] = segment_counter
        print(f"✓ Generated {segment_counter:,} GT-based segments for {split_name} split")
    
    # Return all generated data and statistics for downstream processing
    return segment_files, segment_counts, sorted(list(all_unique_labels)), label_counts_per_split

def save_data_preparation_summary(segment_files, segment_counts, unique_labels, label_counts_per_split, splits):
    """
    Generate comprehensive data preparation summary report in human-readable format.
    
    Creates a detailed text report containing dataset statistics, label distributions,
    file paths, and configuration parameters. This report is essential for
    reproducibility, data validation, and model performance analysis.
    
    Report Sections:
    1. Configuration parameters (window size, step, audio settings)
    2. Input/output file paths for traceability
    3. Segment counts per data split
    4. Label frequency distributions with percentages
    5. Data balance analysis across train/validation/test splits
    
    Parameters:
    ----------
    segment_files (dict): 
        Generated JSONL file paths mapped by split name
    segment_counts (dict): 
        Total segments generated per data split
    unique_labels (list): 
        All voice type labels discovered in dataset
    label_counts_per_split (dict): 
        Label frequency counts organized by data split
    splits (dict):
        Participant-based data splits with file information including participant IDs
    """
    # Create timestamped summary file for tracking multiple preparation runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BasePaths.LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = BasePaths.LOGGING_DIR / f"split_distribution_audio_{timestamp}.txt"

    # Collect input file information for reproducibility tracking
    input_files = {
        'audio_files_dir': str(AudioClassification.CHILDLENS_AUDIO_DIR),
        'participant_info_csv': str(AudioClassification.CHILDLENS_PARTICIPANT_INFO),
        'annotations_dir': str(Path(AudioClassification.CHILDLENS_PARTICIPANT_INFO).parent)
    }

    # Calculate label distribution percentages for balance analysis
    # This helps identify potential class imbalance issues early
    label_percentages_dict = {}
    for split_name, counts in label_counts_per_split.items():
        total_labels_in_split = sum(counts.values())
        label_percentages_dict[split_name] = {
            label: (count / total_labels_in_split * 100) if total_labels_in_split > 0 else 0
            for label, count in counts.items()
        }

    # Calculate the duration of the longest GT segment across all splits
    longest_gt_duration = 0.0
    for split_file in segment_files.values():
        try:
            with open(split_file, 'r') as f:
                for line in f:
                    seg = json.loads(line)
                    duration = seg.get('duration', 0.0)
                    if duration > longest_gt_duration:
                        longest_gt_duration = duration
        except Exception as e:
            print(f"Warning: Could not read {split_file} for longest GT segment calculation: {e}")

    # Create text summary
    lines = []
    lines.append("=== Data Preparation Summary ===")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("Configuration:")
    lines.append(f"  Sample rate: {AudioConfig.SR}")
    lines.append(f"  N mels: {AudioConfig.N_MELS}")
    lines.append(f"  Hop length: {AudioConfig.HOP_LENGTH}")
    lines.append("")

    lines.append(f"Longest GT segment duration: {longest_gt_duration:.2f} seconds")
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

    # Extract unique participant IDs for each split
    split_participant_ids = {}
    split_mapping = {'dev': 'val', 'train': 'train', 'test': 'test'}

    for original_split_name, files_in_split in splits.items():
        split_name = split_mapping.get(original_split_name, original_split_name)
        participant_ids = set()
        for file_info in files_in_split:
            if file_info.get('participant_id'):
                participant_ids.add(file_info['participant_id'])
        split_participant_ids[split_name] = sorted(list(participant_ids))

    # ID Distribution
    lines.append("ID Distribution:")
    for split_name in ['train', 'val', 'test']:
        if split_name in split_participant_ids:
            ids = split_participant_ids[split_name]
            lines.append(f"  {split_name.title()} IDs: {len(ids)}, {ids}")
    lines.append("")

    # ID Overlap Check
    lines.append("ID Overlap Check:")
    train_ids = set(split_participant_ids.get('train', []))
    val_ids = set(split_participant_ids.get('val', []))
    test_ids = set(split_participant_ids.get('test', []))

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        if train_val_overlap:
            lines.append(f"  Train-Val overlap: {sorted(list(train_val_overlap))}")
        if train_test_overlap:
            lines.append(f"  Train-Test overlap: {sorted(list(train_test_overlap))}")
        if val_test_overlap:
            lines.append(f"  Val-Test overlap: {sorted(list(val_test_overlap))}")
    else:
        lines.append("  Overlap found: No")
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
    Create participant ID-based data splits to prevent data leakage in longitudinal studies.
    
    This function implements a duration-balanced splitting strategy that ensures:
    1. No participant appears in multiple data splits (prevents overfitting)
    2. Data splits maintain target proportions (80% train, 10% val, 10% test)
    3. Total duration is balanced across splits for fair evaluation
    4. Files from the same participant remain together (preserves individual patterns)
    
    Splitting Algorithm:
    - Load participant mapping from ChildLens metadata CSV
    - Group annotation files by participant ID  
    - Calculate total duration per participant for balanced allocation
    - Sort participants by duration (largest first for greedy assignment)
    - Assign participants to splits based on duration targets
    - Generate file lists grouped by split assignment
    
    Data Leakage Prevention:
    This approach is critical for longitudinal studies where multiple recordings
    exist per participant. Random file-level splits would allow the model to
    learn participant-specific patterns, leading to inflated performance metrics.
    
    Returns:
    -------
    dict: Participant-based data splits
        Keys: 'train', 'dev', 'test'
        Values: Lists of file info dictionaries with paths and metadata
        
    Raises:
    ------
    FileNotFoundError: If participant mapping CSV is not found
    ValueError: If no valid annotations files are discovered
    """
    # Load participant ID mapping from ChildLens metadata
    id_mapping_df = pd.read_csv(AudioClassification.CHILDLENS_PARTICIPANT_INFO, sep=';')
    file_name_to_id = dict(zip(id_mapping_df['file_name'].astype(str), 
                               id_mapping_df['child_id'].astype(str)))

    print(f"Loaded {len(file_name_to_id)} video-to-ID mappings")

    # Group annotation files by participant ID to prevent data leakage
    id_to_files = defaultdict(list)
    files_without_id = []

    # Discover all JSON annotation files in the metadata directory
    annotations_dir = Path(AudioClassification.CHILDLENS_PARTICIPANT_INFO).parent
    json_files = glob(f"{annotations_dir}/*.json")
    
    # Process each annotation file and extract metadata
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
                
            # Calculate total annotation duration for this file
            duration = max(ann.get('endTime', 0) for ann in annotations)
            
            # Extract base filename and map to participant ID using helper function
            file_name = extract_file_name_from_video_name(video_name)
            participant_id = file_name_to_id.get(file_name, None)
            
            # Create file metadata record
            file_info = {
                "path": json_file,
                "uri": video_name,
                "file_name": file_name,
                "duration": duration,
                "participant_id": participant_id
            }
            
            # Group files by participant ID or track orphaned files
            if participant_id:
                id_to_files[participant_id].append(file_info)
            else:
                files_without_id.append(file_info)
                print(f"Warning: No ID found for file_name '{file_name}' from video '{video_name}'")
                
        except Exception as e:
            print(f"Skipping file {json_file} due to error: {e}")

    print(f"\nFound {len(id_to_files)} unique participant IDs")

    # Calculate total duration per participant for duration-balanced splitting
    id_durations = {}
    for participant_id, files in id_to_files.items():
        total_duration = sum(f["duration"] for f in files)
        id_durations[participant_id] = total_duration

    # Sort participants by total duration (descending) for greedy assignment
    # This ensures largest participants are assigned first for better balance
    sorted_ids = sorted(id_durations.keys(), key=lambda x: id_durations[x], reverse=True)

    # Duration-balanced splitting with target proportions (80/10/10)
    total_duration = sum(id_durations.values())
    target_train = 0.8 * total_duration   # 80% for training
    target_dev = 0.1 * total_duration     # 10% for validation  
    target_test = 0.1 * total_duration    # 10% for testing

    # Initialize split collections and duration trackers
    train_ids, dev_ids, test_ids = [], [], []
    train_duration, dev_duration, test_duration = 0, 0, 0

    # Greedy assignment: assign each participant to split with greatest need
    for participant_id in sorted_ids:
        duration = id_durations[participant_id]
        
        # Calculate remaining duration needed for each split
        train_need = max(0, target_train - train_duration)
        dev_need = max(0, target_dev - dev_duration)
        test_need = max(0, target_test - test_duration)
        
        # Assign participant to split with highest remaining need
        if train_need >= dev_need and train_need >= test_need:
            train_ids.append(participant_id)
            train_duration += duration
        elif dev_need >= test_need:
            dev_ids.append(participant_id)
            dev_duration += duration
        else:
            test_ids.append(participant_id)
            test_duration += duration

    # Convert participant-based splits to file-based splits for downstream processing
    train_files = [f for pid in train_ids for f in id_to_files[pid]]
    dev_files = [f for pid in dev_ids for f in id_to_files[pid]]
    test_files = [f for pid in test_ids for f in id_to_files[pid]]

    splits = {
        "train": train_files,
        "dev": dev_files,
        "test": test_files
    }

    return splits

def main():
    """
    Execute complete audio data preparation pipeline for multi-label voice classification.
    
    This is the main orchestration function that coordinates all data preparation steps
    for the ChildLens longitudinal study audio classification project. The pipeline
    implements best practices for longitudinal data handling, preventing data leakage
    through participant-based splitting.
    
    Pipeline Stages:
    1. Participant ID Mapping: Load participant metadata and create ID-based groupings
    2. Duration-Balanced Splitting: Allocate participants to train/val/test (80/10/10)
    3. Annotation Processing: Parse JSON files and map events to voice classifications
    4. Sliding Window Generation: Create fixed-duration segments with configurable overlap
    5. Multi-Label Assignment: Handle overlapping speech events within time windows
    6. JSONL Export: Generate machine learning ready files for efficient batch loading
    7. Statistical Reporting: Create comprehensive data distribution summaries
    
    Data Leakage Prevention:
    - Participant-based splits ensure no individual appears in multiple sets
    - Critical for longitudinal studies with repeated recordings per child
    - Prevents model overfitting to individual speech patterns
    
    Configuration Sources:
    - AudioConfig: Window parameters, sample rates, valid event types
    - AudioClassification: File paths for input data and participant mapping
    - BasePaths: Output directories for processed data and logs
    
    Output Files:
    - train_segments_w{duration}_s{step}.jsonl: Training data segments
    - val_segments_w{duration}_s{step}.jsonl: Validation data segments  
    - test_segments_w{duration}_s{step}.jsonl: Test data segments
    - split_distribution_audio_{timestamp}.txt: Comprehensive statistics report
    
    Error Handling:
    - Validates audio file availability before processing
    - Handles missing participant IDs gracefully with warnings
    - Ensures minimum segment threshold to prevent empty datasets
    - Provides detailed error messages for debugging
    
    Raises:
    ------
    FileNotFoundError: If required input directories or files are missing
    ValueError: If no valid training segments can be created
    Exception: For any other processing errors with detailed context
    """
    try:
        print("🚀 Starting Streamlined Audio Data Preparation Pipeline")
        print("=" * 60)
        
        # Ensure output directory exists for generated segment files
        output_dir = Path(AudioClassification.INPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Stage 1: Create participant-based splits to prevent data leakage
        print("👥 Creating participant-based train/dev/test splits...")
        splits = create_participant_splits()

        # Stage 2: Generate JSONL training segments from annotation files (MODIFIED CALL)
        print("\n📊 Creating JSONL segments from annotations...")
        segment_files, segment_counts, unique_labels, label_counts_per_split = create_jsonl_segments_from_annotations(
            splits,
            AudioConfig.VALID_RTTM_CLASSES,
            output_dir
        )

        # Validate that segments were successfully created
        total_segments = sum(segment_counts.values())
        if total_segments == 0:
            raise ValueError("No training segments were created. Check your annotations and audio files.")
        
        # Stage 3: Generate comprehensive data preparation summary report
        save_data_preparation_summary(segment_files, segment_counts, unique_labels, label_counts_per_split, splits)
        
        print(f"\n✅ Streamlined data preparation pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Data preparation pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()