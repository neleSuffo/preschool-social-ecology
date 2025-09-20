import argparse
from constants import AudioClassification
from utils import load_thresholds, load_model, process_audio_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio classifier on a folder of audio files.")
    parser.add_argument('--audio_folder', type=str, required=True, help="Path to folder containing WAV audio files.")
    args = parser.parse_args()
    
    model, mlb = load_model()
    if model is None:
        exit()
    
    # Load optimized thresholds from training
    thresholds = load_thresholds(mlb.classes_)
    
    # Process all audio files in the specified folder and save results
    process_audio_folder(args.audio_folder, model, mlb, thresholds)