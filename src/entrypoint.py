#!/usr/bin/env python3
"""
Main entrypoint for the naturalistic social analysis pipeline.

This script runs the complete analysis pipeline:
1. Runs inference on the input video to detect persons, faces, and audio
2. Generates frame-level social interaction classifications
3. Creates interaction segments with validation and merging
4. Produces an annotated video with overlay information
5. Stores all outputs in a structured output folder

Usage:
    python entrypoint.py <video_file_path>
    
Example:
    python entrypoint.py "/Users/nelesuffo/Promotion/ProcessedData/videos_example/quantex_at_home_id255706_2022_04_12_01.MP4"
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
import fire

# Add src directory to path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from inference.main import main as run_model_inference
from results.rq_01.frame_level_analysis import run_frame_level_analysis
from results.rq_01.video_level_analysis import create_interaction_segments
from results.rq_01.create_annotated_video import main as create_annotated_video
from constants import DataPaths, ResearchQuestions

def run_inference(video_path, temp_db_path):
    """
    Run the inference pipeline on the video to detect persons, faces, and audio.
    
    Args:
        video_path (Path): Path to input video
        temp_db_path (Path): Path to temporary database for results
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("STEP 1: RUNNING INFERENCE PIPELINE")
    print("="*60)
    
    try:
        # Import and run the main inference script
        # You'll need to create/adapt this based on your actual inference pipeline
        print(f"🔍 Running inference on: {video_path.name}")
        print(f"📊 Results will be stored in: {temp_db_path}")
        
        # For now, create a placeholder inference function
        # Replace this with your actual inference pipeline
        success = run_model_inference(video_path, temp_db_path)

        if success:
            print("✅ Inference completed successfully")
            return True
        else:
            print("❌ Inference failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def generate_frame_level_analysis(temp_db_path, output_dir):
    """
    Generate frame-level social interaction classifications.
    
    Args:
        temp_db_path (Path): Path to temporary database with inference results
        output_dir (Path): Output directory for results
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("STEP 2: GENERATING FRAME-LEVEL ANALYSIS")
    print("="*60)
    
    try:        
        print(f"📊 Generating frame-level classifications...")
        run_frame_level_analysis(temp_db_path, output_dir)

        if (output_dir / ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV.name).exists():
            print("✅ Frame-level analysis completed successfully")
            return True
        else:
            print("❌ Frame-level analysis failed - output file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during frame-level analysis: {e}")
        return False

def generate_segment_analysis(output_dir):
    """
    Generate interaction segments with validation and merging.
    
    Args:
        output_dir (Path): Output directory for results
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("STEP 3: GENERATING SEGMENT ANALYSIS")
    print("="*60)
    
    try:
        # Import and run segment analysis
        sys.path.append(str(src_path / "results" / "rq_01"))
        from video_level_analysis import create_interaction_segments
        
        # Update paths to use our custom output directory
        original_frame_csv = ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV
        original_segments_csv = ResearchQuestions.INTERACTION_SEGMENTS_CSV
        
        ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV = output_dir / "frame_level_social_interactions.csv"
        ResearchQuestions.INTERACTION_SEGMENTS_CSV = output_dir / "interaction_segments.csv"
        
        print(f"📊 Generating interaction segments...")
        create_interaction_segments()
        
        # Restore original paths
        ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV = original_frame_csv
        ResearchQuestions.INTERACTION_SEGMENTS_CSV = original_segments_csv
        
        if (output_dir / "interaction_segments.csv").exists():
            print("✅ Segment analysis completed successfully")
            return True
        else:
            print("❌ Segment analysis failed - output file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during segment analysis: {e}")
        return False

def create_annotated_video(video_path, output_dir):
    """
    Create annotated video with overlay information.
    
    Args:
        video_path (Path): Path to input video
        output_dir (Path): Output directory for results
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("STEP 4: CREATING ANNOTATED VIDEO")
    print("="*60)
    
    try:
        # Import and run video annotation
        sys.path.append(str(src_path / "results" / "rq_01"))
        from create_annotated_video import create_annotated_video_from_csv
        
        # Update paths to use our custom output directory
        original_frame_csv = ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV
        original_segments_csv = ResearchQuestions.INTERACTION_SEGMENTS_CSV
        original_output_dir = ResearchQuestions.OUTPUT_BASE_DIR
        
        ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV = output_dir / "frame_level_social_interactions.csv"
        ResearchQuestions.INTERACTION_SEGMENTS_CSV = output_dir / "interaction_segments.csv"
        ResearchQuestions.OUTPUT_BASE_DIR = output_dir
        
        print(f"🎬 Creating annotated video for: {video_path.name}")
        result = create_annotated_video_from_csv(video_path)
        
        # Restore original paths
        ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV = original_frame_csv
        ResearchQuestions.INTERACTION_SEGMENTS_CSV = original_segments_csv
        ResearchQuestions.OUTPUT_BASE_DIR = original_output_dir
        
        if result is not None:
            print("✅ Annotated video created successfully")
            return True
        else:
            print("❌ Video annotation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during video annotation: {e}")
        return False

def cleanup_temp_files(temp_db_path):
    """
    Clean up temporary files.
    
    Args:
        temp_db_path (Path): Path to temporary database
    """
    try:
        if temp_db_path.exists():
            temp_db_path.unlink()
            print(f"🧹 Cleaned up temporary database: {temp_db_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up temporary file {temp_db_path}: {e}")

def main(input_path):
    """
    Run the complete naturalistic social analysis pipeline.
    
    Args:
        input_path (str): Path to either:
            - A single video file to process
            - A folder containing multiple video files to process
        
    Example:
        # Single video file
        python entrypoint.py "/path/to/video.mp4"
        
        # Folder with multiple videos
        python entrypoint.py "/path/to/videos_folder/"
        
        # Specific example
        python entrypoint.py "/Users/nelesuffo/Promotion/ProcessedData/videos_example/"
    """
    print("🚀 NATURALISTIC SOCIAL ANALYSIS PIPELINE")
    print("="*60)
    
    # Validate input
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
        
        print(f"🎥 Processing single video: {input_path.name}")
        process_single_video(input_path)
        
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
        
        process_multiple_videos(video_files)
        
    else:
        print(f"❌ Error: {input_path} is neither a file nor a directory")

def process_single_video(video_path):
    """
    Process a single video file through the complete pipeline.
    
    Args:
        video_path (Path): Path to the video file
    """
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_path.stem
    output_dir = Path(f"output_interaction_analysis_{video_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary database path
    temp_db_path = output_dir / "temp_inference.db"
    
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"🎥 Processing video: {video_path.name}")
    
    success = run_pipeline_for_video(video_path, output_dir, temp_db_path)
    
    if success:
        print(f"✅ Successfully processed: {video_path.name}")
        show_results_summary(output_dir)
    else:
        print(f"❌ Failed to process: {video_path.name}")

def process_multiple_videos(video_files):
    """
    Process multiple video files through the complete pipeline.
    
    Args:
        video_files (list): List of Path objects for video files
    """
    # Create batch output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = Path(f"batch_output_interaction_analysis_{timestamp}")
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(sorted(video_files), 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        print(f"{'='*60}")
        
        # Create individual output directory for each video
        video_name = video_file.stem
        video_output_dir = batch_output_dir / f"{video_name}_output"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary database path
        temp_db_path = video_output_dir / "temp_inference.db"
        
        print(f"📁 Video output directory: {video_output_dir.absolute()}")
        
        try:
            success = run_pipeline_for_video(video_file, video_output_dir, temp_db_path)
            if success:
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
    print(f"📁 Batch output directory: {batch_output_dir.absolute()}")
    print(f"Total videos: {len(video_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    if successful > 0:
        print(f"🎉 Batch processing completed with {successful} successful analyses!")
        print(f"📂 Individual video results are in subdirectories of: {batch_output_dir}")

def run_pipeline_for_video(video_path, output_dir, temp_db_path):
    """
    Run the complete pipeline for a single video.
    
    Args:
        video_path (Path): Path to video file
        output_dir (Path): Output directory for results
        temp_db_path (Path): Path to temporary database
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Step 1: Run inference
        if not run_inference(video_path, temp_db_path):
            print("❌ Pipeline failed at inference step")
            return False
        
        # Step 2: Generate frame-level analysis
        if not generate_frame_level_analysis(temp_db_path, output_dir):
            print("❌ Pipeline failed at frame-level analysis step")
            return False
        
        # Step 3: Generate segment analysis
        if not generate_segment_analysis(output_dir):
            print("❌ Pipeline failed at segment analysis step")
            return False
        
        # Step 4: Create annotated video
        if not create_annotated_video(video_path, output_dir):
            print("❌ Pipeline failed at video annotation step")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in pipeline: {e}")
        return False
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_db_path)

def show_results_summary(output_dir):
    """
    Show a summary of the generated results.
    
    Args:
        output_dir (Path): Output directory containing results
    """
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 All outputs saved to: {output_dir.absolute()}")
    print(f"📊 Generated files:")
    
    output_files = list(output_dir.glob("*"))
    for file in sorted(output_files):
        if file.name != "temp_inference.db":
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   • {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    fire.Fire(main)