import os
import shutil
from glob import glob

# Paths
base_path = "/home/nele_pauline_suffo/ProcessedData/face_det_input"
image_dirs = {
    "train": os.path.join(base_path, "images", "train"),
    "val": os.path.join(base_path, "images", "val"),
    "test": os.path.join(base_path, "images", "test"),
}
label_dirs = {
    "train": os.path.join(base_path, "labels", "train"),
    "val": os.path.join(base_path, "labels", "val"),
    "test": os.path.join(base_path, "labels", "test"),
}

# Child IDs for val and test
validation_ids = ['id257291', 'id261609', 'id257573']
test_ids = ['id262565', 'id262691', 'id255706']

# Frames for 5 minutes at 30 FPS = 5 * 60 * 30 = 9000 frames
# Since we're sampling every 10th frame, we want all frames up to frame number 9000
max_frame_number = 5 * 60 * 30  # 9000

def get_first_video_for_id(child_id, src_split):
    """Get the first video name for a given child ID in the source split."""
    # Pattern to match files for this child ID: quantex_at_home_idXXXXXX_YYYY_MM_DD_VV_NNNNNN.PNG/.jpg
    pattern_png = os.path.join(image_dirs[src_split], f"quantex_at_home_{child_id}_*.PNG")
    pattern_jpg = os.path.join(image_dirs[src_split], f"quantex_at_home_{child_id}_*.jpg")
    all_files = glob(pattern_png) + glob(pattern_jpg)
    
    if not all_files:
        return None
    
    # Extract unique video names (everything before the frame number)
    video_names = set()
    for file_path in all_files:
        filename = os.path.basename(file_path)
        # Remove extension first
        filename_no_ext = filename.replace('.PNG', '').replace('.jpg', '')
        # Split by underscore and take everything except the last part (frame number)
        parts = filename_no_ext.split('_')
        if len(parts) >= 6:  # quantex_at_home_idXXXXXX_YYYY_MM_DD_VV_NNNNNN
            video_name = '_'.join(parts[:-1])  # Everything except frame number
            video_names.add(video_name)
    
    if not video_names:
        return None
    
    # Return the first video name alphabetically (this will give us the earliest date/video)
    return sorted(video_names)[0]

def move_first_frames(ids, src_split):
    for child_id in ids:
        # Get the first video for this child ID
        first_video = get_first_video_for_id(child_id, src_split)
        
        if not first_video:
            print(f"No files found for {child_id} in {src_split}")
            continue
        
        # Get all image files for this specific video (both PNG and jpg)
        pattern_png = os.path.join(image_dirs[src_split], f"{first_video}_*.PNG")
        pattern_jpg = os.path.join(image_dirs[src_split], f"{first_video}_*.jpg")
        image_files = sorted(glob(pattern_png) + glob(pattern_jpg))

        if not image_files:
            print(f"No files found for video {first_video} in {src_split}")
            continue

        # Filter files to only include frames up to 5 minutes (frame 9000)
        filtered_files = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Extract frame number from filename (last part before extension)
            filename_no_ext = filename.replace('.PNG', '').replace('.jpg', '')
            frame_number = int(filename_no_ext.split('_')[-1])
            
            if frame_number <= max_frame_number:
                filtered_files.append(img_path)
            else:
                break  # Since files are sorted, we can break early
        
        if not filtered_files:
            print(f"No files within first 5 minutes found for video {first_video} in {src_split}")
            continue

        moved_count = 0
        for img_path in filtered_files:
            filename = os.path.basename(img_path)
            # Handle both extensions for label files
            if filename.endswith('.PNG'):
                label_path = os.path.join(label_dirs[src_split], filename.replace(".PNG", ".txt"))
            else:  # .jpg
                label_path = os.path.join(label_dirs[src_split], filename.replace(".jpg", ".txt"))

            # Destination paths
            dest_img = os.path.join(image_dirs["train"], filename)
            dest_label = os.path.join(label_dirs["train"], os.path.basename(label_path))

            # Move image
            if os.path.exists(img_path):
                shutil.move(img_path, dest_img)
                moved_count += 1
                
            # Move label if exists
            if os.path.exists(label_path):
                shutil.move(label_path, dest_label)

        print(f"Moved {moved_count} images (and labels if present) for {child_id} (video: {first_video}) from {src_split} to train")

def move_back_to_original_splits():
    """Move files back from train to their original val and test directories based on child IDs."""
    
    # Get all files currently in train directory
    train_images_png = glob(os.path.join(image_dirs["train"], "*.PNG"))
    train_images_jpg = glob(os.path.join(image_dirs["train"], "*.jpg"))
    all_train_images = train_images_png + train_images_jpg
    
    moved_to_val = 0
    moved_to_test = 0
    
    for img_path in all_train_images:
        filename = os.path.basename(img_path)
        
        # Extract child ID from filename
        # Format: quantex_at_home_idXXXXXX_YYYY_MM_DD_VV_NNNNNN.ext
        if "quantex_at_home_" in filename:
            # Find the id part
            parts = filename.split('_')
            child_id = None
            for part in parts:
                if part.startswith('id') and len(part) == 8:  # id + 6 digits
                    child_id = part
                    break
            
            if not child_id:
                print(f"Could not extract child ID from {filename}")
                continue
            
            # Determine destination based on child ID
            if child_id in validation_ids:
                dest_split = "val"
                moved_to_val += 1
            elif child_id in test_ids:
                dest_split = "test"
                moved_to_test += 1
            else:
                # This file doesn't belong to val or test IDs, leave it in train
                continue
            
            # Move image file
            dest_img = os.path.join(image_dirs[dest_split], filename)
            shutil.move(img_path, dest_img)
            
            # Move corresponding label file if it exists
            if filename.endswith('.PNG'):
                label_filename = filename.replace('.PNG', '.txt')
            else:  # .jpg
                label_filename = filename.replace('.jpg', '.txt')
            
            label_path = os.path.join(label_dirs["train"], label_filename)
            if os.path.exists(label_path):
                dest_label = os.path.join(label_dirs[dest_split], label_filename)
                shutil.move(label_path, dest_label)
    
    print(f"Moved {moved_to_val} files back to validation")
    print(f"Moved {moved_to_test} files back to test")

# Move files back to their original locations
#print("Moving files back from train to val and test...")
#move_back_to_original_splits()

# Move from validation and test
move_first_frames(validation_ids, "val")
move_first_frames(test_ids, "test")

print("Done!")