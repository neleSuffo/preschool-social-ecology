from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# Assuming you have a frame from your egocentric video as a NumPy array
# For demonstration, let's load a sample image
img_path = '/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id254922_2022_06_29_01/quantex_at_home_id254922_2022_06_29_01_023290.jpg'
img = cv2.imread(img_path)
detections = RetinaFace.detect_faces(img)

# Draw the detections on the image
for face in detections.values():
    # Get bounding box coordinates
    facial_area = face['facial_area']
    x1, y1, x2, y2 = facial_area
    
    # Draw a rectangle for the face
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw the landmarks
    landmarks = face['landmarks']
    for point in landmarks.values():
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

# save the image with detections
output_path = '/home/nele_pauline_suffo/outputs/face_retinaface/face_detection_output.jpg'
cv2.imwrite(output_path, img)