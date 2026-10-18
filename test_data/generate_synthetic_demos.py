"""
PipeGen Demo Video Generator
Uses photorealistic living-room benchmark frames to generate realistic video sequences:
1. demo_normal_activity.mp4: Standing, walking, sitting in chair (0 falls)
2. demo_fall.mp4: Standing, rapid slip & collapse, fallen horizontal posture (1 FALL CONFIRMED)
3. demo_fall_and_recovery.mp4: Standing -> Fall -> Recovery standing back up (1 Fall + Recovery)
"""

import cv2
import numpy as np
from pathlib import Path

def blend_frames(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """Blends two images smoothly."""
    return cv2.addWeighted(img1, 1.0 - alpha, img2, alpha, 0)

def generate_photorealistic_demos():
    images_dir = Path(__file__).parent / "images"
    standing_path = images_dir / "standing.jpg"
    sitting_path = images_dir / "sitting.jpg"
    fallen_path = images_dir / "fallen.jpg"

    if not (standing_path.exists() and sitting_path.exists() and fallen_path.exists()):
        print("[!] Warning: Photorealistic benchmark images missing in test_data/images/")
        return

    # Load images at standard 1280x960 resolution
    img_standing = cv2.imread(str(standing_path))
    img_sitting = cv2.imread(str(sitting_path))
    img_fallen = cv2.imread(str(fallen_path))

    h, w = 720, 1280
    img_standing = cv2.resize(img_standing, (w, h))
    img_sitting = cv2.resize(img_sitting, (w, h))
    img_fallen = cv2.resize(img_fallen, (w, h))

    out_dir = Path(__file__).parent
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25

    # -------------------------------------------------------------
    # 1. Normal Activities Video (Standing -> Sitting in armchair)
    # -------------------------------------------------------------
    normal_video_path = out_dir / "demo_normal_activity.mp4"
    out_normal = cv2.VideoWriter(str(normal_video_path), fourcc, fps, (w, h))
    
    # 40 frames standing/walking
    for _ in range(40):
        out_normal.write(img_standing)
    # 15 frames transition to sitting
    for i in range(15):
        alpha = (i + 1) / 15.0
        out_normal.write(blend_frames(img_standing, img_sitting, alpha))
    # 45 frames sitting upright in armchair
    for _ in range(45):
        out_normal.write(img_sitting)
    out_normal.release()
    print(f"[+] Generated: {normal_video_path.name} (100 frames)")

    # -------------------------------------------------------------
    # 2. Controlled Emergency Fall Video (Standing -> Fall -> Persistent)
    # -------------------------------------------------------------
    fall_video_path = out_dir / "demo_fall.mp4"
    out_fall = cv2.VideoWriter(str(fall_video_path), fourcc, fps, (w, h))
    
    # 35 frames standing
    for _ in range(35):
        out_fall.write(img_standing)
    # 10 frames sudden rapid slip/fall transition
    for i in range(10):
        alpha = (i + 1) / 10.0
        out_fall.write(blend_frames(img_standing, img_fallen, alpha))
    # 65 frames persistent fallen recumbent on floor
    for _ in range(65):
        out_fall.write(img_fallen)
    out_fall.release()
    print(f"[+] Generated: {fall_video_path.name} (110 frames)")

    # -------------------------------------------------------------
    # 3. Fall & Recovery Video (Standing -> Fall -> Recover -> Standing)
    # -------------------------------------------------------------
    recovery_video_path = out_dir / "demo_fall_and_recovery.mp4"
    out_rec = cv2.VideoWriter(str(recovery_video_path), fourcc, fps, (w, h))
    
    # 30 frames standing
    for _ in range(30):
        out_rec.write(img_standing)
    # 10 frames fall transition
    for i in range(10):
        alpha = (i + 1) / 10.0
        out_rec.write(blend_frames(img_standing, img_fallen, alpha))
    # 45 frames fallen (confirms fall + debounce)
    for _ in range(45):
        out_rec.write(img_fallen)
    # 15 frames standing back up transition
    for i in range(15):
        alpha = (i + 1) / 15.0
        out_rec.write(blend_frames(img_fallen, img_standing, alpha))
    # 35 frames recovered standing upright
    for _ in range(35):
        out_rec.write(img_standing)
    out_rec.release()
    print(f"[+] Generated: {recovery_video_path.name} (135 frames)")

if __name__ == "__main__":
    generate_photorealistic_demos()
