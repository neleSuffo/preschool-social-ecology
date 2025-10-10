import os
import cv2
from gfpgan import GFPGANer
from tqdm import tqdm

input_dir = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id255944_2022_03_08_01"
output_dir = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_enhanced"
os.makedirs(output_dir, exist_ok=True)

restorer = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

files = [fname for fname in os.listdir(input_dir) if fname.lower().endswith((".png", ".jpg", ".jpeg"))]
for fname in tqdm(files, desc="Enhancing faces"):
    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)
    
    img = cv2.imread(in_path)
    if img is None:
        continue

    try:
        _, _, restored_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        cv2.imwrite(out_path, restored_img)
    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")