"""
PipeGen Stage 5: Privacy-Preserving Filter
Applies real-time facial anonymization via adaptive Gaussian blurring and pixelation.
Operates strictly at the edge to ensure zero identifiable facial biometrics escape the pipeline.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class PrivacyFilter:
    def __init__(
        self,
        enabled: bool = True,
        blur_kernel_size: int = 35,
        pixelation_factor: int = 14,
        expand_ratio: float = 1.35
    ):
        self.enabled = enabled
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.pixelation_factor = pixelation_factor
        self.expand_ratio = expand_ratio

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def apply(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Applies privacy obfuscation over the facial region of each detected person.
        detections: list of dicts containing 'bbox' [x1, y1, x2, y2] and 'keypoints' np.ndarray (17, 3).
        """
        if not self.enabled or frame is None or len(detections) == 0:
            return frame

        annotated_frame = frame.copy()
        h_frame, w_frame = annotated_frame.shape[:2]

        for det in detections:
            bbox = det.get("bbox")
            kpts = det.get("keypoints")

            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            face_box = None

            # Attempt 1: Accurate keypoint-based facial bounding
            if kpts is not None and len(kpts) >= 5:
                # Keypoints 0..4: nose, l_eye, r_eye, l_ear, r_ear
                head_kpts = kpts[:5]
                valid_kpts = [k for k in head_kpts if k[2] > 0.30]

                if len(valid_kpts) >= 2:
                    xs = [k[0] for k in valid_kpts]
                    ys = [k[1] for k in valid_kpts]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    cx = (min_x + max_x) / 2.0
                    cy = (min_y + max_y) / 2.0
                    face_w = max(max_x - min_x, bw * 0.22) * self.expand_ratio
                    face_h = max(max_y - min_y, bh * 0.20) * self.expand_ratio

                    fx1 = int(max(0, cx - face_w / 2.0))
                    fy1 = int(max(0, cy - face_h / 2.0))
                    fx2 = int(min(w_frame, cx + face_w / 2.0))
                    fy2 = int(min(h_frame, cy + face_h / 2.0))
                    face_box = (fx1, fy1, fx2, fy2)

            # Attempt 2: Fallback to anatomical upper head region
            if face_box is None:
                fx1 = int(max(0, x1 + bw * 0.25))
                fy1 = int(max(0, y1))
                fx2 = int(min(w_frame, x2 - bw * 0.25))
                fy2 = int(min(h_frame, y1 + bh * 0.25))
                face_box = (fx1, fy1, fx2, fy2)

            fx1, fy1, fx2, fy2 = face_box
            if fx2 > fx1 and fy2 > fy1:
                roi = annotated_frame[fy1:fy2, fx1:fx2]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    # 1. Pixelate (Mosaic downscale & upscale)
                    pix_w = max(1, roi.shape[1] // self.pixelation_factor)
                    pix_h = max(1, roi.shape[0] // self.pixelation_factor)
                    small = cv2.resize(roi, (pix_w, pix_h), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # 2. Gaussian Blur over pixelated ROI for smooth non-reversible privacy
                    blurred = cv2.GaussianBlur(pixelated, (self.blur_kernel_size, self.blur_kernel_size), 0)

                    annotated_frame[fy1:fy2, fx1:fx2] = blurred

                    # Subtle privacy indicator border
                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (180, 220, 180), 1)

        return annotated_frame
