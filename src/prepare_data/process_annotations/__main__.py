import os
import argparse
from prepare_data.process_annotations.create_database import (
    write_xml_to_database,
    create_child_class_in_db,
)
from prepare_data.process_annotations.convert_annotations import main as convert_annotations
from constants import VALID_TARGETS

def setup_database() -> None:
    """
    Sets up the database by writing XML data and creating child classes.
    """
    try:
        os.environ['OMP_NUM_THREADS'] = '20'
        print("Setting up the database...")
        write_xml_to_database()
        create_child_class_in_db()
        print("Database setup complete.")
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        raise

def run_yolo_conversion(target: str) -> None:
    """
    Converts annotations to YOLO format for the specified target.

    Parameters
    ----------
    target : str
        Target YOLO label.
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"Invalid target '{target}'. Must be one of: {VALID_TARGETS}")
    
    try:
        os.environ['OMP_NUM_THREADS'] = '20'
        print(f"Starting YOLO conversion for target: {target}...")
        convert_annotations(target)
        print(f"YOLO conversion for target {target} complete.")
    except Exception as e:
        print(f"Error during YOLO conversion for target {target}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process annotations: setup database or convert to YOLO format.")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for setting up the database
    parser_setup_db = subparsers.add_parser('setup_db', help='Set up the annotation database.')
    
    # Subparser for converting to YOLO
    parser_convert_yolo = subparsers.add_parser('convert_yolo', help='Convert annotations to YOLO format.')
    parser_convert_yolo.add_argument(
        '--target', 
        type=str, 
        required=True,
        choices=list(VALID_TARGETS),
        help=f'Target YOLO label. Choose from: {", ".join(VALID_TARGETS)}'
    )

    args = parser.parse_args()

    if args.command == 'setup_db':
        setup_database()
    elif args.command == 'convert_yolo':
        run_yolo_conversion(target=args.target)
    else:
        parser.print_help()