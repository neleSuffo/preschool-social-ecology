import subprocess
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from constants import Inference

# --- Configuration ---
FRAME_ANALYSIS_SCRIPT = Path("results/01_frame_level_analysis.py")
SEGMENT_CREATION_SCRIPT = Path("results/01_video_level_analysis.py")
EVALUATION_SCRIPT = Path("results/eval_segment_performance.py")

def run_command(cmd, step_name):
    """Executes a subprocess command and handles errors."""
    print(f"\n--- Starting: {step_name} ---")
    try:
        # Run command and capture output (needed for the folder path in step 1)
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent, # Adjust CWD if necessary
            encoding='utf-8'
        )
        print(f"✅ {step_name} completed successfully.")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {step_name}: Command failed with return code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ERROR: Python or script not found. Check your PATH and script location.")
        sys.exit(1)

def main(rules=None, plot=False):
    """
    Runs the full pipeline: frame-level analysis, segment creation, and evaluation.
    
    Parameters:
    ----------
    rules : list of int, optional
        List of rule numbers to override default rules in frame-level analysis.
    plot : bool, optional
        Whether to generate plots during evaluation.
    """    
    # 1. --- STEP 1: FRAME-LEVEL ANALYSIS (Generates Timestamped Folder) ---
    frame_cmd = [sys.executable, str(FRAME_ANALYSIS_SCRIPT)]
    if rules:
        frame_cmd.extend(["--rules"] + [str(r) for r in rules])
    
    # We expect the last line of stdout to be the path to the newly created run directory
    frame_output = run_command(frame_cmd, "Frame-Level Analysis")
    
    # Find the timestamped directory path in the output logs of step 1
    # We look for the folder path structure created by rq_01_frame_level_analysis.py
    path_match = re.search(r'interaction_analysis_\d{8}_\d{6}', frame_output)
    
    if path_match:
        # Construct the full path using the base directory
        run_folder = Inference.BASE_OUTPUT_DIR / path_match.group(0)
    else:
        print(f"❌ ERROR: Could not find timestamped output folder in logs. Frame analysis output:\n{frame_output}")
        sys.exit(1)
    
    print(f"➡️ Captured Run Folder: {run_folder}")

    # 2. --- STEP 2: SEGMENT CREATION (Uses Timestamped Folder) ---
    segment_cmd = [
        sys.executable, 
        str(SEGMENT_CREATION_SCRIPT),
        "--folder_path", str(run_folder)
    ]
    run_command(segment_cmd, "Segment Creation")

    # 3. --- STEP 3: EVALUATION & PLOTTING (Uses Timestamped Folder) ---
    
    evaluation_cmd = [
        sys.executable, 
        str(EVALUATION_SCRIPT),
        "--folder_path", str(run_folder),
    ]
    if plot:
        evaluation_cmd.append("--plot")
    run_command(evaluation_cmd, "Evaluation and Plotting")
    
    print("\n\n🎉 Full Pipeline Execution Complete!")
    print(f"Final results saved in: {run_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full multimodal analysis pipeline.")
    parser.add_argument("--rules", type=int, nargs='+', help="Override default rule set (e.g., 1 2 3 4 5).")
    parser.add_argument("--plot", action='store_true', help="Generate plots during evaluation.")
    args = parser.parse_args()
    
    main(rules=args.rules, plot=args.plot)