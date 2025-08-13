import subprocess
import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List
from prepare_data.prepare_dataset import main as prepare_dataset
from prepare_data.process_videos import main as process_videos
from prepare_data.crop_detections import main as crop_detections
from constants import VALID_TARGETS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    """Handles the data preparation pipeline steps."""
    
    def __init__(self, target: Optional[str], threads: int = 2):
        self.target = target
        os.environ['OMP_NUM_THREADS'] = str(threads)
            
    def prepare_dataset(self):
        """Prepare training dataset."""
        if not self.target:
            raise ValueError("YOLO target is required to prepare the dataset.")
        logging.info(f"Preparing training dataset for {self.target}...")
        prepare_dataset(self.target)
        
    def run_pipeline(self, steps: List[str]):
        """Run specified pipeline steps in order."""
        step_mapping = {
            'videos': self.process_videos,
            'setup_annotations_db': self.setup_annotation_database,
            'annotations': self.convert_annotations_and_crop, # This now covers YOLO conversion and cropping
            'dataset': self.prepare_dataset
        }
        
        for step in steps:
            if step in step_mapping:
                try:
                    step_mapping[step]()
                except Exception as e:
                    logging.error(f"Error in {step} step: {e}")
                    raise # Stop pipeline on error
            else:
                logging.warning(f"Unknown step: {step}")

def main():
    parser = argparse.ArgumentParser(description='Data preparation pipeline')
    parser.add_argument('--dataset', action='store_true', help='Prepare training dataset')
    parser.add_argument('--target', type=str, choices=VALID_TARGETS, 
                        help='Target YOLO label for annotations and dataset preparation. Required for annotations and dataset steps.')
    parser.add_argument('--all', action='store_true', help='Run full pipeline (videos, setup_annotations_db, annotations, dataset)')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads for OMP_NUM_THREADS')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.target, args.threads)
    
    steps_to_run = []
    canonical_order = ['videos', 'setup_annotations_db', 'annotations', 'dataset']

    if args.all:
        if not args.target:
            parser.error("--target is required when --all is specified.")
        steps_to_run = canonical_order
    else:
        if args.dataset:
            if not args.target:
                parser.error("--target is required when --dataset is specified.")
            steps_to_run.append('dataset')
        
        final_ordered_steps = []
        for step_name in canonical_order:
            if step_name in steps_to_run and step_name not in final_ordered_steps:
                final_ordered_steps.append(step_name)
        steps_to_run = final_ordered_steps
        
    if not steps_to_run:
        # This means no action flags were set, or target was missing for a required action (which parser.error would have caught)
        if not (args.videos or args.setup_annotations_db or args.annotations or args.dataset or args.all):
            parser.error("No pipeline steps specified. Use --videos, --setup_annotations_db, --annotations, --dataset, or --all.")
        # If steps_to_run is empty but some flags were set, it implies an issue already handled by parser.error
        return


    if steps_to_run:
        pipeline.run_pipeline(steps_to_run)
    else:
        # This case should ideally not be reached if the above logic is correct
        logging.info("No steps to run.")


if __name__ == "__main__":
    main()