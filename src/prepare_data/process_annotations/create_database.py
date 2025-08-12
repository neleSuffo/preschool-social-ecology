import sqlite3
import json
import cv2
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from constants import DataPaths
from config import LabelMapping

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def write_xml_to_database() -> None:
    """
    This function creates a SQLite database from the annotations XML file.
    """
    logging.info("Starting to create database annotations.")

    # Create the directory if it does not exist
    Path(DataPaths.ANNO_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    added_videos = set()
    added_images = set()
    added_categories = set()
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()

        # Drop tables if they exist
        cursor.execute("DROP TABLE IF EXISTS annotations")
        cursor.execute("DROP TABLE IF EXISTS videos")
        cursor.execute("DROP TABLE IF EXISTS images")
        cursor.execute("DROP TABLE IF EXISTS categories")

        # Create tables for annotations and videos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                image_id INTEGER,
                video_id INTEGER,
                category_id INTEGER,
                bbox TEXT,
                outside INTEGER,
                person_visibility INTEGER,
                person_ID INTEGER,
                person_age TEXT,
                person_gender TEXT,
                gaze_directed_at_child TEXT,
                object_interaction BOOLEAN
            )
        """)

        cursor.execute("""
             CREATE TABLE IF NOT EXISTS images (
                 video_id INTEGER,
                 frame_id INTEGER,
                 file_name TEXT
             )
         """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY,
                category TEXT,
                supercategory TEXT
            )
        """)

        # Commit and close the database connection
        conn.commit()
        for file_name in DataPaths.ANNO_DIR.iterdir():
            if file_name.suffix == '.xml':
                add_annotations_to_db(cursor, conn, file_name, added_images, added_videos, added_categories)
                
        logging.info("Database setup complete.")

def add_annotations_to_db(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    xml_path: Path,
    added_images: set,
    added_videos: set,
    added_categories: set
) -> None:
    """ 
    This function adds annotations from an XML file to the database.
    
    Parameters
    ----------
    cursor : sqlite3.Cursor
        the cursor object
    conn : sqlite3.Connection
        the connection object
    xml_path : Path
        the path to the XML file
    added_images : set
        a set of added images
    added_videos : set
        a set of added videos
    added_categories : set
        a set of added categories
    """
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Navigate to the <task> element
    task_element = root.find('meta/task')

    # Extract <name> values
    task_name = task_element.find('name').text
    
    # Add video details if not already added
    if task_name not in added_videos:
        cursor.execute(
            "INSERT INTO videos (file_name) VALUES (?)",
            (f"{task_name}.mp4",)
        )
        video_id = cursor.lastrowid
        added_videos.add(task_name)
    else:
        # Get the video id if it was already added
        cursor.execute("SELECT id FROM videos WHERE file_name = ?", (f"{task_name}.mp4",))
        video_id = cursor.fetchone()[0]
        
    # Iterate over all 'track' elements
    for track in root.iter("track"):
        track_label = track.get("label")

        # Map the label to its corresponding label id using the dictionary
        # returns -1 if the label is not in the dictionary
        track_label_id = LabelMapping.LABEL_TO_ID_MAPPING.get(track_label, -1)
        # Map the label to its corresponding supercategory using the dictionary
        supercategory = LabelMapping.ID_TO_SUPERCATEGORY_MAPPING.get(
            track_label_id, LabelMapping.unknown_supercategory
        )  # returns "unknown" if the label is not in the dictionary

        # Add category details if not already added
        if track_label_id not in added_categories:
            cursor.execute(
            """
            INSERT INTO categories (id, category, supercategory)
            VALUES (?, ?, ?)
            """,
                (
                track_label_id,
                track_label,
                supercategory,
                ),
            )
            added_categories.add(track_label_id)
    
        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            row = box.attrib
            outside = int(row["outside"]) # 0 if the object is inside the frame, 1 if it is outside
            frame_id_padded = f'{int(row["frame"]):06}'
            bbox_json = json.dumps([float(row["xtl"]), float(row["ytl"]), float(row["xbr"]), float(row["ybr"])])
            
            # Extract other attributes
            person_visibility = box.find(".//attribute[@name='Visibility']")
            person_visibility_value = int(person_visibility.text) if person_visibility is not None else None

            person_id = box.find(".//attribute[@name='ID']")
            person_id_value = int(person_id.text) if person_id is not None else None

            person_age = box.find(".//attribute[@name='Age']")
            person_age_value = person_age.text if person_age is not None else None

            person_gender = box.find(".//attribute[@name='Gender']")
            person_gender_value = person_gender.text if person_gender is not None else None#
            
            gaze_directed_at_child = box.find(".//attribute[@name='Gaze Directed at Child']")
            gaze_directed_at_child_value = gaze_directed_at_child.text if gaze_directed_at_child is not None else None

            object_interaction = box.find(".//attribute[@name='Interaction']")
            object_interaction_value = object_interaction.text if object_interaction is not None else "No"

            # Insert the annotation into the database
            cursor.execute(
                """
                INSERT INTO annotations (image_id, video_id, category_id, bbox, outside, person_visibility, person_ID, person_age, person_gender, gaze_directed_at_child, object_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row["frame"], # image_id
                    video_id, # video_id
                    track_label_id, # category_id
                    bbox_json, # bbox coordinates
                    outside,
                    person_visibility_value,
                    person_id_value,
                    person_age_value,
                    person_gender_value,
                    gaze_directed_at_child_value,
                    object_interaction_value,
                ),
            )
            # Add image details if not already added
            image_name = f"{task_name}_{frame_id_padded}.jpg"
            if image_name not in added_images:
                cursor.execute(
                """
                    INSERT INTO images (video_id, frame_id, file_name)
                    VALUES (?, ?, ?)
                """,
                    (
                    video_id,
                    row["frame"], # frame_id
                    image_name,
                    ),
                ) 
                added_images.add(image_name)
    conn.commit()
    logging.info(f'Database commit for file {xml_path.name} successful')

def create_child_class_in_db():
    """
    This function creates a new class "child_body part" in the database.
    """
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    cursor = conn.cursor()
    # Update the category_id for child body parts
    query_1 = """
    UPDATE annotations
    SET category_id = 11
    WHERE category_id = 1 AND person_ID = 1;
    """
    # Add new category to the categories table
    query_2 = """
    INSERT INTO categories (id, category, supercategory)
    VALUES (11, 'child_body_part', 'person');
    """
    # Execute the query
    cursor.execute(query_1)
    logging.info("Successfully updated category_id for child body parts.")
    cursor.execute(query_2)
    logging.info("Successfully added new category 'child_body_part' to the categories table.")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
