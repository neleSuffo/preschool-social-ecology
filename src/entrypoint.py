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
from results.rq_01.frame_level_analysis import main as run_frame_level_analysis
from results.rq_01.video_level_analysis import main as create_interaction_segments
from results.rq_01.create_annotated_video import main as create_annotated_video
from constants import DataPaths, Inference

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
        
        success = run_model_inference(video_path=video_path, db_path=temp_db_path)

        if success:
            print("✅ Inference completed successfully")
            return True
        else:
            print("❌ Inference failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def generate_frame_level_analysis(temp_db_path: Path, output_dir: Path):
    """
    Generate frame-level social interaction classifications.
    
    Parameters
    ----------
    temp_db_path (Path)
        Path to temporary database with inference results
    output_dir (Path)
        Output directory for results

    Returns
    -------
    bool: 
        True if successful, False otherwise
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
        create_interaction_segments(output_dir)

        file_name = Inference.INTERACTION_SEGMENTS_CSV.name
        if (output_dir / file_name).exists():
            print("✅ Segment analysis completed successfully")
            return True
        else:
            print("❌ Segment analysis failed - output file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during segment analysis: {e}")
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

def process_single_video(video_path: Path):
    """
    Process a single video file through the complete pipeline.
    
    Parameters
    ----------
    video_path (Path)
        Path to the video file
    """
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_path.stem
    output_dir = Path(f"output_interaction_analysis_{video_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary database path
    temp_db_path = output_dir / "temp_inference.db"
    
    success = run_pipeline_for_video(video_path, output_dir, temp_db_path)
    
    if success:
        print(f"✅ Successfully processed: {video_path.name}")
        show_results_summary(output_dir)
    else:
        print(f"❌ Failed to process: {video_path.name}")

def process_multiple_videos(video_files, input_dir):
    """
    Process multiple video files through the complete pipeline with shared analysis.
    
    Args:
        video_files (list): List of Path objects for video files
        input_dir (Path): Input directory containing the videos
    """
    # Create single output directory named after input directory
    output_dir = Path(f"output_{input_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary database path for all videos
    temp_db_path = output_dir / "temp_inference.db"
    
    print(f"📁 Single output directory: {output_dir.absolute()}")
    print(f"🎬 Processing {len(video_files)} videos with shared analysis...")
    
    try:
        # Step 1: Run inference for ALL videos (only once)
        print(f"\n{'='*60}")
        print("STEP 1: RUNNING INFERENCE FOR ALL VIDEOS")
        print(f"{'='*60}")
        
        inference_success = True
        for i, video_file in enumerate(sorted(video_files), 1):
            print(f"🔍 Running inference {i}/{len(video_files)} on: {video_file.name}")
            if not run_inference(video_file, temp_db_path):
                print(f"❌ Inference failed for: {video_file.name}")
                inference_success = False
                break
        
        if not inference_success:
            print("❌ Pipeline failed during inference step")
            return
        
        # Step 2: Generate frame-level analysis for ALL videos (only once)
        print(f"\n{'='*60}")
        print("STEP 2: GENERATING FRAME-LEVEL ANALYSIS FOR ALL VIDEOS")
        print(f"{'='*60}")
        
        if not generate_frame_level_analysis(temp_db_path, output_dir):
            print("❌ Pipeline failed at frame-level analysis step")
            return
        
        # Step 3: Generate segment analysis for ALL videos (only once)
        print(f"\n{'='*60}")
        print("STEP 3: GENERATING SEGMENT ANALYSIS FOR ALL VIDEOS")
        print(f"{'='*60}")
        
        if not generate_segment_analysis(output_dir):
            print("❌ Pipeline failed at segment analysis step")
            return
        
        # Step 4: Create annotated videos for each video individually
        print(f"\n{'='*60}")
        print("STEP 4: CREATING ANNOTATED VIDEOS")
        print(f"{'='*60}")
        
        successful_videos = 0
        failed_videos = 0
        
        for i, video_file in enumerate(sorted(video_files), 1):
            print(f"\n🎬 Creating annotated video {i}/{len(video_files)}: {video_file.name}")
            
            try:
                if create_annotated_video_for_shared_analysis(video_file, output_dir):
                    successful_videos += 1
                    print(f"✅ Successfully created annotated video: {video_file.name}")
                else:
                    failed_videos += 1
                    print(f"❌ Failed to create annotated video: {video_file.name}")
            except Exception as e:
                failed_videos += 1
                print(f"❌ Error creating annotated video for {video_file.name}: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"Total videos: {len(video_files)}")
        print(f"✅ Successful annotated videos: {successful_videos}")
        print(f"❌ Failed annotated videos: {failed_videos}")
        
        if successful_videos > 0:
            print(f"🎉 Batch processing completed!")
            show_shared_results_summary(output_dir, successful_videos)
    
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_db_path)

def create_annotated_video_for_shared_analysis(video_path, output_dir):
    """
    Create annotated video using shared CSV files from the output directory.
    
    Args:
        video_path (Path): Path to the video file
        output_dir (Path): Output directory containing shared CSV files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if required CSV files exist
        frame_csv = output_dir / "frame_level_social_interactions.csv"
        segments_csv = output_dir / "interaction_segments.csv"
        
        if not frame_csv.exists():
            print(f"❌ Frame-level CSV not found: {frame_csv}")
            return False
        
        if not segments_csv.exists():
            print(f"❌ Segments CSV not found: {segments_csv}")
            return False
        
        # Create annotated video in the shared output directory
        annotated_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
        
        # Import the annotation function
        from results.rq_01.create_annotated_video import create_annotated_video_with_custom_csvs
        
        print(f"🎬 Creating annotated video: {annotated_video_path.name}")
        
        success = create_annotated_video_with_custom_csvs(
            video_path=str(video_path),
            frame_csv_path=str(frame_csv),
            segments_csv_path=str(segments_csv),
            output_path=str(annotated_video_path)
        )
        
        if success and annotated_video_path.exists():
            print(f"✅ Annotated video created: {annotated_video_path.name}")
            return True
        else:
            print(f"❌ Failed to create annotated video: {annotated_video_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating annotated video for {video_path.name}: {e}")
        return False

def show_shared_results_summary(output_dir, successful_videos):
    """
    Show a summary of the shared analysis results.
    
    Args:
        output_dir (Path): Output directory containing results
        successful_videos (int): Number of successfully processed videos
    """
    print(f"📁 All outputs saved to: {output_dir.absolute()}")
    print(f"📊 Generated files:")
    
    # Show CSV files
    frame_csv = output_dir / "frame_level_social_interactions.csv"
    segments_csv = output_dir / "interaction_segments.csv"
    
    if frame_csv.exists():
        size_mb = frame_csv.stat().st_size / (1024 * 1024)
        print(f"   📋 {frame_csv.name} ({size_mb:.1f} MB) - Combined frame-level analysis")
    
    if segments_csv.exists():
        size_mb = segments_csv.stat().st_size / (1024 * 1024)
        print(f"   📋 {segments_csv.name} ({size_mb:.1f} MB) - Combined segment analysis")
    
    # Show annotated videos
    annotated_videos = list(output_dir.glob("*_annotated.mp4"))
    print(f"   🎬 {len(annotated_videos)} annotated videos:")
    for video in sorted(annotated_videos):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"      • {video.name} ({size_mb:.1f} MB)")

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
            return False
        
        # Step 2: Generate frame-level analysis
        if not generate_frame_level_analysis(temp_db_path, output_dir):
            return False
        
        # Step 3: Generate segment analysis
        if not generate_segment_analysis(output_dir):
            return False
        
        # Step 4: Create annotated video
        if not create_annotated_video(video_path, output_dir):
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
            
def main(input_path: str):
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
        
        process_multiple_videos(video_files, input_path)
        
    else:
        print(f"❌ Error: {input_path} is neither a file nor a directory")

if __name__ == "__main__":
    fire.Fire(main)