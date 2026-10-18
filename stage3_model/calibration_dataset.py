"""
PipeGen Stage 3: INT8 Calibration Dataset Generator
Generates representative calibration batches for TensorRT IInt8EntropyCalibrator2.
"""

import cv2
import numpy as np
from pathlib import Path

def generate_calibration_dataset(num_samples: int = 100):
    dataset_dir = Path(__file__).parent / "calibration_data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Generating {num_samples} INT8 calibration image frames in: {dataset_dir.resolve()}")

    # Generate synthetic diverse elderly room background + human silhouette poses
    for i in range(num_samples):
        # Create a realistic room background (walls, floor, furniture)
        img = np.ones((640, 640, 3), dtype=np.uint8) * 220 # Light wall color
        
        # Floor line
        floor_y = 480
        cv2.rectangle(img, (0, floor_y), (640, 640), (140, 150, 160), -1)
        
        # Add varied illumination & noise
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw human figure in diverse postures: standing, sitting, bending, lying
        pose_type = i % 4
        center_x = int(np.random.uniform(180, 460))
        
        if pose_type == 0:
            # Standing posture (upright)
            head_y = int(np.random.uniform(120, 180))
            cv2.circle(img, (center_x, head_y), 24, (60, 60, 60), -1)
            cv2.line(img, (center_x, head_y + 24), (center_x, head_y + 180), (70, 70, 70), 20)
            cv2.line(img, (center_x, head_y + 180), (center_x - 30, floor_y), (80, 80, 80), 14)
            cv2.line(img, (center_x, head_y + 180), (center_x + 30, floor_y), (80, 80, 80), 14)

        elif pose_type == 1:
            # Sitting posture (chair)
            head_y = int(np.random.uniform(220, 270))
            cv2.circle(img, (center_x, head_y), 24, (60, 60, 60), -1)
            cv2.line(img, (center_x, head_y + 24), (center_x, head_y + 140), (70, 70, 70), 20)
            cv2.line(img, (center_x, head_y + 140), (center_x + 60, head_y + 140), (80, 80, 80), 14)
            cv2.line(img, (center_x + 60, head_y + 140), (center_x + 60, floor_y), (80, 80, 80), 14)

        elif pose_type == 2:
            # Bending posture
            head_y = int(np.random.uniform(260, 320))
            cv2.circle(img, (center_x - 70, head_y), 24, (60, 60, 60), -1)
            cv2.line(img, (center_x - 70, head_y), (center_x, head_y + 100), (70, 70, 70), 20)
            cv2.line(img, (center_x, head_y + 100), (center_x, floor_y), (80, 80, 80), 14)

        else:
            # Fallen / horizontal posture
            head_x = center_x - 120
            head_y = floor_y - 20
            cv2.circle(img, (head_x, head_y), 22, (60, 60, 60), -1)
            cv2.line(img, (head_x + 22, head_y), (center_x + 80, head_y), (70, 70, 70), 20)
            cv2.line(img, (center_x + 80, head_y), (center_x + 140, head_y), (80, 80, 80), 14)

        filename = dataset_dir / f"calib_frame_{i:04d}.jpg"
        cv2.imwrite(str(filename), img)

    print(f"[+] Successfully generated {num_samples} calibration frames for TensorRT INT8 PTQ.")
    return dataset_dir

if __name__ == "__main__":
    generate_calibration_dataset(50)
