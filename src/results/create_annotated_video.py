# Research Question 1. How much time do children spend alone?
#
# This script creates an annotated video by overlaying interaction segments on the original video.

import cv2
import sys
import pandas as pd
import subprocess
import shutil
import fire
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import ResearchQuestions
from config import DataConfig

# Constants
FPS = DataConfig.FPS # frames per second

def create_annotated_video_from_csv(video_path: Path, final_output_dir: Path):
    """
    Annotate a video and create a final video file by combining
    annotated frames with the original audio using FFMPEG.
    
    Parameters
    ----------
    video_path (str/Path): 
        Path to the input video file or directory containing video files.
    output_dir (str/Path): 
        Path to the output directory where the annotated video will be saved.
    """
    print("=" * 60)
    print("CREATING ANNOTATED VIDEO WITH FFMPEG")
    print("=" * 60)
    
    # Set paths to existing CSV files
    segments_csv_path = ResearchQuestions.INTERACTION_SEGMENTS_CSV
    frame_csv_path = ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV
    
    # Define temporary and final output directories
    temp_frames_dir = Path("temp_annotated_frames")
    
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
                'has_kchi': row.get('has_kchi', 0),
                'has_ohs': row.get('has_ohs', 0),
                'has_cds': row.get('has_cds', 0),
                'rule1_turn_taking': row.get('rule1_turn_taking', 0),
                'rule2_close_proximity': row.get('rule2_close_proximity', 0),
                'rule3_cds_speaking': row.get('rule3_cds_speaking', 0),
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
            if current_detection['has_kchi']:
                speech_info.append("KCHI")
            if current_detection['has_ohs']:
                speech_info.append("OHS")
            if current_detection['has_cds']:
                speech_info.append("CDS")
            speech_text = f"Speech: {', '.join(speech_info) if speech_info else 'Silent'}"
            speech_color = (0, 255, 255) if speech_info else text_color  # Yellow if speech
            cv2.putText(frame, speech_text, (10, y_pos), font, font_scale, speech_color, 1)
            y_pos += line_height
            
            # Interaction rules (active rules)
            active_rules = []
            if current_detection['rule1_turn_taking']:
                active_rules.append("1:Turn-Taking")
            if current_detection['rule2_close_proximity']:
                active_rules.append("2:Close-Prox")
            if current_detection['rule3_cds_speaking']:
                active_rules.append("3:CDS")
            if current_detection['rule4_person_recent_speech']:
                active_rules.append("4:Person+Speech")
            
            rules_text = f"Rules: {', '.join(active_rules) if active_rules else 'None'}"
            rules_color = (0, 255, 0) if active_rules else text_color  # Green if any rules active
            cv2.putText(frame, rules_text, (300, y_pos), font, font_scale, rules_color, 1)
            
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

def main(input_path, final_output_dir: Path = ResearchQuestions.OUTPUT_BASE_DIR):
    """
    Create annotated video(s) from existing segments and frame-level data.
    
    Args:
        input_path (str): Path to either:
            - A single video file to annotate
            - A folder containing multiple video files to process
    
    Example:
        # Single video file
        python create_annotated_video.py "/path/to/video.mp4"
        
        # Folder with multiple videos
        python create_annotated_video.py "/path/to/videos_folder/"
        
        # Specific example
        python create_annotated_video.py "/Users/nelesuffo/Promotion/ProcessedData/videos_example/"
    """
    if not final_output_dir:
        print("❌ Error: Please provide a final output directory")
        return

    final_output_dir = Path(final_output_dir)
    
    if not input_path:
        print("❌ Error: Please provide an input path")
        print("Usage: python create_annotated_video.py <video_path_or_folder>")
        return
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ Error: Path not found at {input_path}")
        return
    
    # Determine if input is a file or directory
    if input_path.is_file():
        # Single file processing
        if not input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.MP4']:
            print(f"❌ Error: Unsupported video file extension: {input_path.suffix}")
            return
        
        result = create_annotated_video_from_csv(input_path, final_output_dir)

        if result is None:
            print(f"❌ Failed to process: {input_path.name}")
            
    elif input_path.is_dir():
        # Directory processing - find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
        video_files = [f for f in input_path.iterdir() 
                    if f.is_file() and f.suffix in video_extensions]
        
        if not video_files:
            print(f"❌ Error: No video files found in {input_path}")
            print(f"Looking for extensions: {', '.join(sorted(video_extensions))}")
            return
        
        print(f"📁 Found {len(video_files)} video files in: {input_path}")
        print(f"🎬 Processing multiple videos...")
        
        successful = 0
        failed = 0
        
        for i, video_file in enumerate(sorted(video_files), 1):
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(video_files)}: {video_file.name}")
            print(f"{'='*60}")
            
            try:
                result = create_annotated_video_from_csv(video_file, final_output_dir)
                if result is not None:
                    successful += 1
                    print(f"✅ Successfully processed: {video_file.name}")
                else:
                    failed += 1
                    print(f"❌ Failed to process: {video_file.name}")
            except Exception as e:
                failed += 1
                print(f"❌ Error processing {video_file.name}: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos: {len(video_files)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if successful > 0:
            print(f"🎉 Batch processing completed with {successful} successful annotations!")
        
    else:
        print(f"❌ Error: {input_path} is neither a file nor a directory")

if __name__ == "__main__":
    fire.Fire(main)