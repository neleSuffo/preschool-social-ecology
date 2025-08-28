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

def main(video_file_path):
    """
    Run the complete naturalistic social analysis pipeline.
    
    Args:
        video_file_path (str): Path to the input video file
        
    Example:
        python entrypoint.py "/path/to/video.mp4"
    """
    print("🚀 NATURALISTIC SOCIAL ANALYSIS PIPELINE")
    print("="*60)
    
    # Validate input
    video_path = Path(video_file_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return

    if not video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.MP4']:
        print(f"⚠️ Warning: Unusual video file extension: {video_path.suffix}")
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_path.stem
    output_dir = Path(f"output_interaction_analysis_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary database path
    temp_db_path = output_dir / "temp_inference.db"
    
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"🎥 Processing video: {video_path.name}")
    
    success_steps = []
    
    try:
        # Step 1: Run inference
        if run_inference(video_path, temp_db_path):
            success_steps.append("inference")
        else:
            print("❌ Pipeline failed at inference step")
            return
        
        # Step 2: Generate frame-level analysis
        if generate_frame_level_analysis(temp_db_path, output_dir):
            success_steps.append("frame_analysis")
        else:
            print("❌ Pipeline failed at frame-level analysis step")
            return
        
        # Step 3: Generate segment analysis
        if generate_segment_analysis(output_dir):
            success_steps.append("segment_analysis")
        else:
            print("❌ Pipeline failed at segment analysis step")
            return
        
        # Step 4: Create annotated video
        if create_annotated_video(video_path, output_dir):
            success_steps.append("video_annotation")
        else:
            print("❌ Pipeline failed at video annotation step")
            return
        
        # Success!
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
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in pipeline: {e}")
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_db_path)

if __name__ == "__main__":
    fire.Fire(main)