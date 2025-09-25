from retinaface import RetinaFace
from pathlib import Path
import pandas as pd
import cv2

# Assuming you have a frame from your egocentric video as a NumPy array
# For demonstration, let's load a sample image
video_name = 'quantex_at_home_id255237_2022_05_08_04'
frame_id = "13080"
frame_id_padded = frame_id.zfill(6)
frame_dir = "/home/nele_pauline_suffo/ProcessedData/face_det_input/images/test"
img_path = f"{frame_dir}/{video_name}_{frame_id_padded}"
# try .PNG or .jpg depending on your files
if Path(img_path + '.jpg').exists():
    img_path = img_path + '.jpg'
    img = cv2.imread(img_path)
elif Path(img_path + '.PNG').exists():
    img_path = img_path + '.PNG'
    img = cv2.imread(img_path)
else:
    raise FileNotFoundError(f"Image not found: {img_path}.jpg or {img_path}.PNG")
detections = RetinaFace.detect_faces(img)

gt_label = pd.read_csv('/home/nele_pauline_suffo/outputs/face_retinaface/ground_truth_data.csv')
gt_label = gt_label[(gt_label['video_name'] == video_name) & (gt_label['frame_id'] == int(frame_id))]

# print how many faces were detected
print(f"Detected {len(detections)} faces in frame {frame_id} of video {video_name}")
# Draw the detections on the image
for face in detections.values():
    # Get bounding box coordinates
    facial_area = face['facial_area']
    x1, y1, x2, y2 = facial_area
    
    # Draw a rectangle for the face
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # draw a second rectangle for ground truth with a different color
    # Assuming you have ground truth coordinates (gt_x1, gt_y1, gt_x2, gt_y2)
    #gt_x1, gt_y1, gt_x2, gt_y2 = gt_label[['x1', 'y1', 'x2', 'y2']].values[0]
    #cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), (255, 0, 0), 2)

    # Draw the landmarks
    landmarks = face['landmarks']
    for point in landmarks.values():
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

# save the image with detections
output_dir = Path('/home/nele_pauline_suffo/outputs/face_retinaface/')
image_name = Path(img_path).name
cv2.imwrite(str(output_dir / image_name), img)

print(f"Detections saved to {output_dir / image_name}")