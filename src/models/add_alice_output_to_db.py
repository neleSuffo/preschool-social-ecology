#!/usr/bin/env python3
"""
Script to parse ALICE output and add KCHI vocalizations to the database.

ALICE output format:
/path/to/file_starttime_endtime.wav    phonemes    syllables    words

Example:
/home/.../quantex_at_home_id254922_2022_04_12_01_0001361360_0001404930.wav    32.09    15.08    10.98
"""

import sqlite3
import argparse
import logging
import re
from pathlib import Path
from typing import Tuple, Optional
from constants import DataPaths, Vocalizations
from config import KchiVoc_Config
from inference.inference import get_video_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_vocalizations_table(cursor: sqlite3.Cursor):
    """
    Create the Vocalizations table if it doesn't exist.
    Also insert the ALICE model into the Models table.
    
    Parameters:
    ----------
    cursor : sqlite3.Cursor
        Database cursor
    """
    # Insert ALICE model into Models table
    cursor.execute('''
        INSERT OR IGNORE INTO Models (model_name, description, output_variables) VALUES 
        ('vocalization', 'ALICE for vocalization analysis',
         '{"phonemes": "float", "syllables": "float", "words": "float"}')
    ''')
    
    # Get the model_id for the ALICE model
    cursor.execute('SELECT model_id FROM Models WHERE model_name = ?', ('vocalization',))
    result = cursor.fetchone()
    alice_model_id = result[0] if result else None
    
    if alice_model_id is None:
        logging.error("Failed to get model_id for kchi_vocalization model")
        raise ValueError("Could not retrieve model_id for ALICE model")
    
    # Create KchiVocalizations table without DEFAULT for model_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Vocalizations (
            vocalization_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            speaker STRING,
            start_time_seconds REAL,
            end_time_seconds REAL,
            phonemes REAL,
            syllables REAL,
            words REAL,
            model_id INTEGER,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')
    logging.info(f"Created/verified Vocalizations table and inserted ALICE model (model_id: {alice_model_id})")
    
    return alice_model_id

def parse_filename(filename: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Parse filename to extract video name, start time, and end time.
    
    Example filename: quantex_at_home_id254922_2022_04_12_01_0001361360_0001404930.wav
    
    Parameters:
    ----------
    filename : str
        Filename to parse
        
    Returns:
    -------
    tuple: (video_name, start_time_seconds, end_time_seconds)
    """
    # Remove file extension
    basename = Path(filename).stem
    
    # Pattern to match: video_name_starttime_endtime
    # Start and end times are typically 10 digits (milliseconds * 100)
    pattern = r'^(.+)_(\d{10})_(\d{10})$'
    match = re.match(pattern, basename)
    
    if match:
        video_name = match.group(1)
        start_time_raw = int(match.group(2))
        end_time_raw = int(match.group(3))
        
        # Convert from centiseconds to seconds
        # Example: 0000737080 -> 737080 -> 73.7080 seconds
        # The timestamp appears to be in units of 1/10000 second
        start_time_seconds = start_time_raw / 10000.0
        end_time_seconds = end_time_raw / 10000.0
        
        return video_name, start_time_seconds, end_time_seconds
    else:
        logging.warning(f"Could not parse filename: {filename}")
        return basename, None, None

def parse_alice_output_line(line: str) -> Tuple[str, float, float, float]:
    """
    Parse a single line of ALICE output.
    
    Parameters:
    ----------
    line : str
        Line from ALICE output file (format: filename phonemes syllables words)
        
    Returns:
    -------
    tuple: (filename, phonemes, syllables, words)
    """
    parts = line.strip().split('\t')
    if len(parts) != 4:
        raise ValueError(f"Expected 4 tab-separated values, got {len(parts)}: {line}")
    
    filename = parts[0]
    phonemes = float(parts[1])
    syllables = float(parts[2])
    words = float(parts[3])
    
    return filename, phonemes, syllables, words

def process_alice_output(alice_file: Path, cursor: sqlite3.Cursor, conn: sqlite3.Connection, alice_model_id: int):
    """
    Process ALICE output file and insert data into database.
    
    Parameters:
    ----------
    alice_file : Path
        Path to ALICE output file
    cursor : sqlite3.Cursor
        Database cursor
    conn : sqlite3.Connection
        Database connection
    alice_model_id : int
        Model ID for the ALICE model
    """
    # Validate input file
    if not alice_file.exists():
        logging.error(f"ALICE output file not found: {alice_file}")
        return
    
    logging.info(f"Processing ALICE output file: {alice_file}")
    
    if 'KCHI' in str(alice_file):
        speaker = 'KCHI'
    elif 'OTH' in str(alice_file):
        speaker = 'FEM_MAL'
    else:
        logging.error(f"Unknown speaker in file: {alice_file}")
        return
    
    processed_count = 0
    skipped_count = 0
    
    with open(alice_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the line
                filename, phonemes, syllables, words = parse_alice_output_line(line)
                
                # Extract video info from filename
                video_name, start_time, end_time = parse_filename(filename)
                
                if start_time is None or end_time is None:
                    logging.warning(f"Line {line_num}: Could not parse times from {filename}")
                    skipped_count += 1
                    continue
                
                # Get video_id
                video_id = get_video_id(video_name, cursor)
                if video_id is None:
                    logging.warning(f"Line {line_num}: Video {video_name} not found in database")
                    skipped_count += 1
                    continue
                
                # Insert into database with explicit model_id
                cursor.execute('''
                    INSERT INTO Vocalizations 
                    (video_id, speaker, start_time_seconds, end_time_seconds, phonemes, syllables, words, model_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (video_id, speaker, start_time, end_time, phonemes, syllables, words, alice_model_id))
                
                processed_count += 1
                
                # Log progress every 100 entries
                if processed_count % 100 == 0:
                    conn.commit()
                
            except Exception as e:
                logging.error(f"Line {line_num}: Error processing line '{line}': {e}")
                skipped_count += 1
                continue
    
    # Final commit
    conn.commit()
    
    logging.info(f"Processing complete:")
    logging.info(f"  Processed: {processed_count} entries")
    logging.info(f"  Skipped: {skipped_count} entries")

def main():
    """
    Main function to process ALICE output and add to database.
    """       
    try:
        # Connect to database
        conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
        cursor = conn.cursor()
        
        # Create table and get model_id
        alice_model_id = create_vocalizations_table(cursor)
        
        # Process ALICE output with the model_id for KCHI and OTH
        process_alice_output(Vocalizations.KCHI_OUTPUT_FILE, cursor, conn, alice_model_id)
        process_alice_output(Vocalizations.OTH_OUTPUT_FILE, cursor, conn, alice_model_id)

        # Close database connection
        conn.close()
        
        logging.info("✅ ALICE output processing completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error processing ALICE output: {e}")
        raise

if __name__ == "__main__":
    main()