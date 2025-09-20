import argparse
from pathlib import Path
from constants import AudioClassification
from utils import load_thresholds, load_inference_model, process_audio_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio classifier on a folder of audio files.")
    parser.add_argument('--audio_folder', type=str, required=True, help="Path to folder containing WAV audio files.")
    args = parser.parse_args()
    
    # Load the trained model and MultiLabelBinarizer once
    results_dir = Path(AudioClassification.RESULTS_DIR)
    model_path = results_dir / "best_model.keras"
    
    model, mlb = load_inference_model(model_path)
    if model is None:
        exit()
    
    # Load optimized thresholds from training
    thresholds = load_thresholds(results_dir, mlb.classes_)
    
    output_dir = AudioClassification.OUTPUT_DIR

    # Process all audio files in the specified folder and save results
    process_audio_folder(args.audio_folder, model, mlb, output_dir, thresholds)