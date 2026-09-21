import subprocess
import sys
import os
import logging
import argparse
from pathlib import Path
from constants import Analysis

# Configure basic logging for the runner
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_command(cmd: list, 
                step_name: str,
                run_folder: Path = None) -> bool:
    """
    Executes a subprocess command and handles output stream logging.
    
    Parameters:
    ----------
    cmd: list
        The command to execute as a list of strings.
    step_name: str
        A descriptive name for the step being executed (used in logs).
    run_folder: Path, optional
        The folder where the command's output is expected to be saved.

    Returns
    -------
    bool
        True if the command executed successfully, False otherwise.
    """
    logging.info(f"\n--- Starting: {step_name} ---")
    # Copy current system environment
    env = os.environ.copy()
    if run_folder:
        # Inject the run_folder into the environment
        env["RUN_FOLDER"] = str(run_folder)
        
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent,
            text=True,
            env=env,
            encoding='utf-8'
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ ERROR in {step_name}: Command failed.")
        sys.exit(1)

def main(social_state_mode='tertiary', 
         video_list=None):
    """
    Runs the streamlined inference pipeline.
    
    Parameters:
    ----------
    social_state_mode (str): 
        'binary' or 'tertiary' classification mode. Default is 'tertiary'.
    video_list (list of str): 
        Optional list of specific video filenames to process (e.g. ['video1.mp4', 'video2.mp4']).
    """
    #1. --- STEP 1: FRAME-LEVEL ANALYSIS ---
    frame_cmd = [
        sys.executable, str(Analysis.FRAME_ANALYSIS_SCRIPT),
        "--social_state_mode", social_state_mode
    ]
    if video_list:
        frame_cmd.extend(["--video_list"] + video_list)

    run_command(frame_cmd, "Step 1: Frame-Level Analysis")
    
    # 2. --- IDENTIFY OUTPUT FOLDER ---
    all_runs = sorted(Analysis.BASE_OUTPUT_DIR.glob("analysis_*"), key=lambda x: x.stat().st_mtime)
    run_folder = all_runs[-1]
    logging.info(f"➡️ Created Run Folder: {run_folder}")
    
    # 3. --- STEP 2: SEGMENT CREATION (CPD Smoothing) ---
    segment_cmd = [
        sys.executable, str(Analysis.SEGMENT_CREATION_SCRIPT),
        "--output_folder_path", str(run_folder),
        "--social_state_mode", social_state_mode
    ]
    run_command(segment_cmd, "Step 2: Video-Level Smoothing", run_folder=run_folder)
    logging.info(f"Final segments saved: {run_folder / Analysis.INTERACTION_SEGMENTS_CSV.name}")

    # 4. --- STEP 3: RESEARCH QUESTION ANALYSIS (Standard vs. Ground Truth) ---
    rq_configs = [
        ("STANDARD PIPELINE", []),
        ("GROUND TRUTH VALIDATION", ["--use_gt"]),
    ]
    
    for label, gt_flag in rq_configs:
        logging.info(f"\n{'='*25} RUNNING RQ SCRIPTS: {label} {'='*25}")
        
        # RQ 02: KCS (Production)
        run_command([sys.executable, "heuristics/rq_02_kcs.py",
                    "--output_folder", str(run_folder)]
                    + gt_flag,
                    f"RQ 02: Child Language Production Analysis ({label})")
        
        # # RQ 03: KCDS (Exposure)
        run_command([sys.executable, "heuristics/rq_03_kcds.py",
                    "--output_folder", str(run_folder)]
                    + gt_flag,
                    f"RQ 03: Child-Directed Speech Exposure Analysis ({label})")

        # RQ 04: Turn-Taking
        run_command([sys.executable, "heuristics/rq_04_turn_taking.py",
                    "--social_state_mode", social_state_mode,
                    "--output_folder", str(run_folder)]
                    + gt_flag,
                    f"RQ 04: Conversational Dynamics ({label})",
                    run_folder=run_folder)
        
        # RQ 05: Interaction Composition
        run_command([sys.executable, "heuristics/rq_05_interaction_composition.py", 
                    "--social_state_mode", social_state_mode,
                    "--output_folder", str(run_folder)]
                    + gt_flag,
                    f"RQ 05: Final Composition ({label})",
                    run_folder=run_folder)
    
    logging.info("\n" + "="*40)
    logging.info("🎉 INFERENCE COMPLETE (Standard & GT generated)")
    logging.info("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Social Analysis Inference Runner")
    parser.add_argument("--social_state_mode", type=str, choices=['binary', 'tertiary'], default='tertiary',
                        help="Classification mode (default: tertiary)")
    parser.add_argument("--video_list", type=str, nargs='+', 
                        help="List of specific videos to process (e.g. video1.mp4 video2.mp4)")
    
    args = parser.parse_args()
    
    # check if args.video_list is a txt file containing video names or a folder containing videos
    if args.video_list and len(args.video_list) == 1 and args.video_list[0].endswith('.txt'):
        with open(args.video_list[0], 'r') as f:
            video_list = [line.strip() for line in f if line.strip()]
        args.video_list = video_list
    # check if args.video_list is a folder containing videos
    elif args.video_list and len(args.video_list) == 1 and Path(args.video_list[0]).is_dir():
        video_folder = Path(args.video_list[0])
        video_list = [str(video) for video in video_folder.iterdir() if video.is_file() and video.suffix.lower() == ".mp4"]
        args.video_list = video_list
    main(social_state_mode=args.social_state_mode, video_list=args.video_list)