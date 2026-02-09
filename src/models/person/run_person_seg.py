import cv2
import numpy as np
from ultralytics import YOLO

# Load the model and image
model = YOLO('/home/nele_pauline_suffo/models/yolo12l_person.pt')
image_path = '/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id254922_2022_05_21_01/quantex_at_home_id254922_2022_05_21_01_034320.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Get original dimensions before running inference (important for resizing masks)
original_height, original_width, _ = image.shape

# Perform inference
# Note: YOLO automatically resizes the image before inference.
results = model(image)

# Process and visualize the first result
result = results[0]

if result.masks is not None:
    # Get the raw mask tensor data (scaled resolution)
    # Shape is typically (N, H_scaled, W_scaled)
    mask_data_scaled = result.masks.data.cpu().numpy() 

    # Iterate over each scaled mask
    for i, mask_scaled in enumerate(mask_data_scaled):
        
        # 1. Resize the scaled mask back to the original image dimensions
        # Use INTER_NEAREST for segmentation masks (boolean data)
        mask_resized = cv2.resize(
            mask_scaled, 
            (original_width, original_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 2. Convert to boolean array using the threshold (0.5 or 0, since it's already float)
        # Using > 0.5 to ensure it's properly thresholded boolean if it wasn't already
        mask_boolean = (mask_resized > 0.5) 
        
        # 3. Create the colored overlay array
        color = [np.random.randint(0, 255) for _ in range(3)]
        mask_image = np.zeros_like(image, dtype=np.uint8)
        
        # 4. Apply the correctly sized boolean mask to the overlay image
        mask_image[mask_boolean] = color
        
        # 5. Add overlay to original image
        image = cv2.addWeighted(image, 1.0, mask_image, 0.5, 0)

    # Display or save the annotated image
    output_path = 'quantex_at_home_id254922_2022_05_21_01_034320_annotated.jpg'
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

else:
    print("No masks found for this image.")