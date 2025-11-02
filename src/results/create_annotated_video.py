# Research Question 1. How much time do children spend alone?
#
# This script creates an annotated video by overlaying interaction segments on the original video.

import cv2
import sys
import pandas as pd
import subprocess
import shutil
import json
import argparse
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Inference
from config import DataConfig, InferenceConfig

# Constants
FPS = DataConfig.FPS # frames per second

def find_frame_level_file(search_dir: Path) -> Path or None:
    """
    Searches the directory for the frame level file.
    
    The pattern assumes files are named like: 
    '02_interaction_segments_<rules>_...csv' or similar.
    """
    # Pattern to look for segment files
    segment_files = list(search_dir.glob(f'{Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem}*{Inference.FRAME_LEVEL_INTERACTIONS_CSV.suffix}'))
    
    if not segment_files:
        return None
    
    segment_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return segment_files[0]

def create_annotated_video_from_csv(video_path: Path, segments_csv_path: Path, frame_csv_path: Path, final_output_dir: Path):
    """
    Annotate a video and create a final video file by combining
    annotated frames with the original audio using FFMPEG.
    
    Parameters
    ----------
    video_path (Path): 
        Path to the input video file.
    segments_csv_path (Path):
        Path to the segment-level interactions CSV (Required).
    frame_csv_path (Path):
        Path to the frame-level interactions CSV (Required).
    final_output_dir (Path): 
        Path to the output directory where the annotated video will be saved.
    """
    print("=" * 60)
    print("CREATING ANNOTATED VIDEO WITH FFMPEG")
    print("=" * 60)
    
    # Define temporary directory
    temp_frames_dir = final_output_dir / "temp_annotated_frames"
    
    # Create necessary directories
    shutil.rmtree(temp_frames_dir, ignore_errors=True) # Clean up old temp directory
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing segments data
    try:
        segments_df = pd.read_csv(segments_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Segments file not found at {segments_csv_path}")
        print("Ensure the correct output folder was specified and segments exist.")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None

    # Get video name from path
    video_name = video_path.stem
    
    # Filter segments for the specific video
    video_segments = segments_df[segments_df['video_name'] == video_name].copy()
    if len(video_segments) == 0:
        print(f"❌ Error: No segments found for video_name '{video_name}' in {segments_csv_path.name}")
        print(f"Available video names in segments: {segments_df['video_name'].unique()[:10]}")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None

    print(f"🎯 Found {len(video_segments)} segments for video '{video_name}'")

    # Load existing frame-level data for detection details
    try:
        frame_data = pd.read_csv(frame_csv_path)
        video_frame_data = frame_data[frame_data['video_name'] == video_name].copy()
    except FileNotFoundError:
        print(f"⚠️ Warning: Frame-level data not found at {frame_csv_path.name}. Using segments only.")
        video_frame_data = pd.DataFrame()
    
    # Check if video file exists
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None
    
    # Get video properties
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure dimensions are even (required by many codecs)
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
        
    frame_number = 0
    
    # Create a lookup for segments by frame
    segment_lookup = {}
    for _, segment in video_segments.iterrows():
        start_frame = int(segment['segment_start'])
        end_frame = int(segment['segment_end'])
        for frame in range(start_frame, end_frame + 1):
            segment_lookup[frame] = {
                'interaction_type': segment['interaction_type'],
                'duration': segment['duration_sec'],
                'start_time': segment['start_time_sec'],
                'end_time': segment['end_time_sec']
            }
    
    # Create frame-level detection lookup
    detection_lookup = {}
    if not video_frame_data.empty:
        for _, row in video_frame_data.iterrows():
            frame_num = int(row['frame_number'])
            detection_lookup[frame_num] = {
                # Note: These column names assume the structure output by rq_01_frame_level_analysis.py
                'person_or_face_present': row.get('person_or_face_present', 0),
                'proximity': row.get('proximity', None),
                'has_kchi': row.get('has_kchi', 0),
                'has_ohs': row.get('has_ohs', 0),
                'has_cds': row.get('has_cds', 0),
                'rule1_turn_taking': row.get('rule1_turn_taking', 0),
                'rule2_close_proximity': row.get('rule2_close_proximity', 0),
                'rule3_cds_speaking': row.get('rule3_kcds_speaking', 0),
                'rule4_person_recent_speech': row.get('rule4_person_recent_speech', 0),
            }
    
    print("🎬 Processing video frames and saving as images...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ensure frame matches expected dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Convert frame number to match our data (aligned to SAMPLE_RATE from config)
        aligned_frame = int(round(frame_number / DataConfig.FPS) * InferenceConfig.SAMPLE_RATE)
        
        # Get current segment info
        current_segment = segment_lookup.get(aligned_frame, {})
        current_detection = detection_lookup.get(aligned_frame, {})
        
        # Calculate overlay area (bottom 120 pixels)
        overlay_height = 120
        overlay_start = height - overlay_height
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, overlay_start), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (255, 255, 255)
        line_height = 25
        
        y_pos = overlay_start + 20
        
        # Line 1: Time information
        current_time = frame_number / fps
        time_text = f"Time: {current_time:.1f}s | Frame: {frame_number} (Aligned: {aligned_frame})"
        cv2.putText(frame, time_text, (10, y_pos), font, font_scale, text_color, 1)
        y_pos += line_height
        
        # Line 2: Segment information
        if current_segment:
            segment_text = f"Segment: {current_segment['interaction_type']} ({current_segment['duration']:.1f}s)"
            # Color code segments
            if current_segment['interaction_type'] == 'Interacting':
                segment_color = (0, 255, 0)  # Green
            elif current_segment['interaction_type'] == 'Available':
                segment_color = (0, 255, 255)  # Yellow
            else:  # Alone
                segment_color = (0, 0, 255)  # Red
            cv2.putText(frame, segment_text, (10, y_pos), font, font_scale, segment_color, 2)
        else:
            cv2.putText(frame, "Segment: No segment", (10, y_pos), font, font_scale, (128, 128, 128), 1)
        
        # Line 2 (Right): Visual Presence
        if current_detection:
            fused_presence = current_detection.get('person_or_face_present', 0)
            prox = current_detection.get('proximity')
            
            presence_text = f"Visual: {'Present' if fused_presence else 'Absent'}"
            if prox is not None:
                 presence_text += f" (Prox: {prox:.2f})"
                 
            presence_color = (0, 255, 0) if fused_presence else (128, 128, 128)
            cv2.putText(frame, presence_text, (300, y_pos), font, font_scale, presence_color, 1)
        y_pos += line_height
        
        # Line 3: Speech detection
        if current_detection:
            speech_info = []
            if current_detection['has_kchi']:
                speech_info.append("KCHI")
            if current_detection['has_ohs']:
                speech_info.append("OHS")
            if current_detection['has_cds']:
                speech_info.append("CDS")
            speech_text = f"Speech: {', '.join(speech_info) if speech_info else 'Silent'}"
            speech_color = (0, 255, 255) if speech_info else text_color  # Yellow if speech
            cv2.putText(frame, speech_text, (10, y_pos), font, font_scale, speech_color, 1)

            # Line 3 (Right): Interaction rules (active rules)
            active_rules = []
            if current_detection.get('rule1_turn_taking'):
                active_rules.append("1:TT")
            if current_detection.get('rule2_close_proximity'):
                active_rules.append("2:Prox")
            if current_detection.get('rule3_cds_speaking'):
                active_rules.append("3:SustCDS")
            if current_detection.get('rule4_person_recent_speech'):
                active_rules.append("4:Vis+RecSpeech")
            
            rules_text = f"Rules: {', '.join(active_rules) if active_rules else 'None'}"
            rules_color = (0, 255, 0) if active_rules else (128, 128, 128) # Green if any rules active
            cv2.putText(frame, rules_text, (300, y_pos), font, font_scale, rules_color, 1)
            y_pos += line_height
            
        # Save the annotated frame to the temporary directory
        image_path = temp_frames_dir / f"frame_{frame_number:05d}.png"
        cv2.imwrite(str(image_path), frame)
        
        frame_number += 1
        
        # Progress indicator
        if total_frames > 0 and frame_number % (30 * 10) == 0: # Every 10 seconds at 30fps
            progress = (frame_number / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    
    # --- FFMPEG Call to combine video and audio ---
    print("\n🚀 Combining frames and original audio with FFMPEG...")
    
    video_from_frames_name = f"{video_name}_annotated_no_audio.mp4"
    final_output_name = f"{video_name}_annotated.mp4"
    
    # Step 1: Combine the images into a video stream
    cmd_video = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', str(temp_frames_dir / 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if it exists
        str(final_output_dir / video_from_frames_name)
    ]
    try:
        subprocess.run(cmd_video, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ Error: FFMPEG not found. Please install it and ensure it's in your system's PATH.")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during FFMPEG video creation: {e}")
        print(f"Stderr: {e.stderr.decode()}")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None
    
    # Step 2: Merge video with original audio
    cmd_audio = [
        'ffmpeg',
        '-i', str(final_output_dir / video_from_frames_name),
        '-i', str(video_path),
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y',  # Overwrite final output
        str(final_output_dir / final_output_name)
    ]
    try:
        subprocess.run(cmd_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Final video with audio saved to: {final_output_dir / final_output_name}")
        
        # Cleanup temporary files
        if (final_output_dir / video_from_frames_name).exists():
            (final_output_dir / video_from_frames_name).unlink()
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
                
        return str(final_output_dir / final_output_name)  # Return path to successful output
        
    except Exception as e:
        print(f"❌ Error during audio merging: {e}")
        # Still cleanup on failure
        if (final_output_dir / video_from_frames_name).exists():
            (final_output_dir / video_from_frames_name).unlink()
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return None

def main(video_path, output_folder: Path):
    """
    Main function to orchestrate the video annotation process.
    
    Parameters
    ---------
    video_path (str): 
        Path to the single video file.
    output_folder (Path): 
        Directory containing the segment and frame analysis files.
    """
    
    # 1. Find the latest segment and frame CSV files in the output folder
    frame_csv_path = find_frame_level_file(output_folder)
    segments_csv_path = Path(output_folder / Inference.INTERACTION_SEGMENTS_CSV)
    
    if not frame_csv_path.exists():
        print(f"⚠️ Warning: Corresponding frame file {frame_csv_path.name} not found. Using segments only for annotation.")
        # Proceed with segments only, the annotation function handles missing frame data gracefully.
    
        
    if segments_csv_path is None:
        print(f"❌ Error: Could not find any segment CSV file in {output_folder}. Ensure analysis has run.")
        return
    
    # 2. Check video file
    video_path = Path(video_path)
    if not video_path.is_file():
        print(f"❌ Error: Video file not found at {video_path}")
        return

    # 3. Create annotated video
    create_annotated_video_from_csv(video_path, segments_csv_path, frame_csv_path, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an annotated video by overlaying segment analysis results.")
    parser.add_argument('--video_path', type=str, help='Path to the single input video file.')
    parser.add_argument('--output_folder', type=str, help='Path to the analysis output folder containing the segment/frame CSVs.')
    
    args = parser.parse_args()
    
    # Use Path objects for consistency
    main(args.video_path, Path(args.output_folder))