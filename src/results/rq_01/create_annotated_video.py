# Research Question 1. How much time do children spend alone?
#
# This script analyzes multimodal social interaction patterns by combining:
# - Visual person detection (child/adult bodies)
# - Visual face detection (child/adult faces with proximity measures)  
# - Audio vocalization detection (child/other speaker identification)
#
# The analysis produces frame-level classifications of social contexts to understand
# when children are alone vs. in various types of social interactions.

import sqlite3
import cv2
import re
import sys
import pandas as pd
import subprocess
import shutil
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, ResearchQuestions
from config import DataConfig, Research_QuestionConfig

# Constants
FPS = DataConfig.FPS # frames per second

def create_annotated_video_from_csv(video_path):
    """
    Annotate a video and create a final video file by combining
    annotated frames with the original audio using FFMPEG.
    
    Parameters
    ----------
    video_path (str/Path): 
        Path to the input video file
    """
    print("=" * 60)
    print("CREATING ANNOTATED VIDEO WITH FFMPEG")
    print("=" * 60)
    
    # Set paths to existing CSV files
    segments_csv_path = ResearchQuestions.INTERACTION_SEGMENTS_CSV
    frame_csv_path = ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV
    
    # Define temporary and final output directories
    temp_frames_dir = Path("temp_annotated_frames")
    final_output_dir = Path(ResearchQuestions.OUTPUT_BASE_DIR)
    
    # Create necessary directories
    shutil.rmtree(temp_frames_dir, ignore_errors=True) # Clean up old temp directory
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing segments data
    try:
        segments_df = pd.read_csv(segments_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Segments file not found at {segments_csv_path}")
        print("Run the main analysis first to generate segments!")
        return None

    # Get video name from path
    video_path = Path(video_path)
    video_name = video_path.stem
    
    # Filter segments for the specific video
    video_segments = segments_df[segments_df['video_name'] == video_name].copy()
    if len(video_segments) == 0:
        print(f"❌ Error: No segments found for video_name '{video_name}'")
        print(f"Available video names in segments: {segments_df['video_name'].unique()[:10]}")
        return None

    print(f"🎯 Found {len(video_segments)} segments for video '{video_name}'")

    # Load existing frame-level data for detection details
    try:
        frame_data = pd.read_csv(frame_csv_path)
        video_frame_data = frame_data[frame_data['video_name'] == video_name].copy()
    except FileNotFoundError:
        print("⚠️ Warning: Frame-level data not found. Using segments only.")
        video_frame_data = pd.DataFrame()
    
    # Check if video file exists
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
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
    
    print(f"🎥 Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    frame_number = 0
    
    # Create a lookup for segments by frame
    segment_lookup = {}
    for _, segment in video_segments.iterrows():
        start_frame = int(segment['segment_start'])
        end_frame = int(segment['segment_end'])
        for frame in range(start_frame, end_frame + 1):
            segment_lookup[frame] = {
                'category': segment['category'],
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
                'child_present': row.get('child_present', 0),
                'adult_present': row.get('adult_present', 0),
                'has_child_face': row.get('has_child_face', 0),
                'has_adult_face': row.get('has_adult_face', 0),
                'proximity': row.get('proximity', None),
                'speaker': row.get('speaker', ''),
                'kchi_speech': 'KCHI' in str(row.get('speaker', '')),
                'other_speech': 'FEM_MAL' in str(row.get('speaker', ''))
            }
    
    print("🎬 Processing video frames and saving as images...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ensure frame matches expected dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Convert frame number to match our data (10-frame intervals)
        aligned_frame = int(round(frame_number / 10) * 10)
        
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
        
        # Time information
        current_time = frame_number / fps
        time_text = f"Time: {current_time:.1f}s | Frame: {frame_number} ({aligned_frame})"
        cv2.putText(frame, time_text, (10, y_pos), font, font_scale, text_color, 1)
        y_pos += line_height
        
        # Segment information
        if current_segment:
            segment_text = f"Segment: {current_segment['category']} ({current_segment['duration']:.1f}s)"
            # Color code segments
            if current_segment['category'] == 'Interacting':
                segment_color = (0, 255, 0)  # Green
            elif current_segment['category'] == 'Co-present Silent':
                segment_color = (0, 255, 255)  # Yellow
            else:  # Alone
                segment_color = (0, 0, 255)  # Red
            cv2.putText(frame, segment_text, (10, y_pos), font, font_scale, segment_color, 2)
        else:
            cv2.putText(frame, "Segment: No segment", (10, y_pos), font, font_scale, (128, 128, 128), 1)
        y_pos += line_height
        
        # Detection information
        if current_detection:
            # Person detection
            person_info = []
            if current_detection['child_present']:
                person_info.append("Child")
            if current_detection['adult_present']:
                person_info.append("Adult")
            person_text = f"Persons: {', '.join(person_info) if person_info else 'None'}"
            cv2.putText(frame, person_text, (10, y_pos), font, font_scale, text_color, 1)
            
            # Face detection with proximity
            face_info = []
            if current_detection['has_child_face']:
                face_info.append("Child Face")
            if current_detection['has_adult_face']:
                face_info.append("Adult Face")
            face_text = f"Faces: {', '.join(face_info) if face_info else 'None'}"
            if current_detection['proximity'] is not None:
                face_text += f" (Prox: {current_detection['proximity']:.2f})"
            cv2.putText(frame, face_text, (300, y_pos), font, font_scale, text_color, 1)
            y_pos += line_height
            
            # Speech detection
            speech_info = []
            if current_detection['kchi_speech']:
                speech_info.append("KCHI")
            if current_detection['other_speech']:
                speech_info.append("Other")
            speech_text = f"Speech: {', '.join(speech_info) if speech_info else 'Silent'}"
            speech_color = (0, 255, 255) if speech_info else text_color  # Yellow if speech
            cv2.putText(frame, speech_text, (10, y_pos), font, font_scale, speech_color, 1)
            
        # Save the annotated frame to the temporary directory
        image_path = temp_frames_dir / f"frame_{frame_number:05d}.png"
        cv2.imwrite(str(image_path), frame)
        
        frame_number += 1
        
        # Progress indicator
        if frame_number % 300 == 0:  # Every 10 seconds at 30fps
            progress = (frame_number / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    
    print(f"✅ All annotated frames saved to: {temp_frames_dir}")
    print(f"📊 Processed {frame_number} frames with {len(video_segments)} segments")

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
        print(f"✅ Video stream created successfully: {final_output_dir / video_from_frames_name}")
    except FileNotFoundError:
        print("❌ Error: FFMPEG not found. Please install it and ensure it's in your system's PATH.")
        shutil.rmtree(temp_frames_dir)
        return
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during FFMPEG video creation: {e}")
        print(f"Stderr: {e.stderr.decode()}")
        shutil.rmtree(temp_frames_dir)
        return
    
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
    except Exception as e:
        print(f"❌ Error during audio merging: {e}")
        shutil.rmtree(temp_frames_dir)
        return

if __name__ == "__main__":
    # Create annotated video from existing CSV
    VIDEO_PATH = "/Users/nelesuffo/Promotion/ProcessedData/videos_example/quantex_at_home_id255944_2022_03_08_01.MP4"
    create_annotated_video_from_csv(VIDEO_PATH)