"""
PipeGen Stage 5: Multi-Person Real-Time Tracker
Associates person detections across consecutive video frames to maintain consistent track IDs.
Mimics NVIDIA NvDCF / ByteTrack behavior for edge inference.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

import math

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def compute_track_affinity(box1: np.ndarray, box2: np.ndarray) -> float:
    """Combines IoU and spatial center proximity to track persons undergoing rapid posture collapse."""
    iou = compute_iou(box1, box2)
    c1 = ((box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0)
    c2 = ((box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0)
    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
    diag = max(100.0, math.hypot(box1[2] - box1[0], box1[3] - box1[1]))
    prox = max(0.0, 1.0 - (dist / (diag * 1.6)))
    return 0.5 * iou + 0.5 * prox

class TrackedPerson:
    def __init__(self, track_id: int, bbox: np.ndarray, keypoints: np.ndarray, confidence: float):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.keypoints = keypoints # (17, 3)
        self.confidence = confidence
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def update(self, bbox: np.ndarray, keypoints: np.ndarray, confidence: float):
        # Exponential smoothing on bounding box to reduce jitter
        alpha = 0.70
        self.bbox = alpha * bbox + (1.0 - alpha) * self.bbox
        self.keypoints = keypoints
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        self.age += 1
        self.time_since_update += 1

class MultiPersonTracker:
    def __init__(self, iou_threshold: float = 0.30, max_age: int = 30, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_id = 1

    def update(self, detections: List[dict]) -> List[dict]:
        """
        detections: list of dicts with keys 'bbox', 'keypoints', 'confidence'.
        Returns updated detections list augmented with persistent 'track_id'.
        """
        if not detections:
            # Mark all existing tracks as missed
            for track in list(self.tracks.values()):
                track.mark_missed()
                if track.time_since_update > self.max_age:
                    del self.tracks[track.track_id]
            return []

        det_boxes = [d["bbox"] for d in detections]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].bbox for tid in track_ids]

        matched_tracks = set()
        matched_dets = set()
        matches = []

        if len(track_boxes) > 0 and len(det_boxes) > 0:
            # Compute affinity matrix (IoU + spatial center proximity)
            iou_matrix = np.zeros((len(det_boxes), len(track_boxes)), dtype=np.float32)
            for d_idx, dbox in enumerate(det_boxes):
                for t_idx, tbox in enumerate(track_boxes):
                    iou_matrix[d_idx, t_idx] = compute_track_affinity(dbox, tbox)

            # Greedy Hungarian-like matching
            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < 0.20:
                    break
                d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matches.append((d_idx, t_idx))
                matched_dets.add(d_idx)
                matched_tracks.add(t_idx)
                iou_matrix[d_idx, :] = -1.0
                iou_matrix[:, t_idx] = -1.0

        # Update matched tracks
        for d_idx, t_idx in matches:
            tid = track_ids[t_idx]
            det = detections[d_idx]
            self.tracks[tid].update(det["bbox"], det["keypoints"], det["confidence"])
            det["track_id"] = tid

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackedPerson(
                    track_id=tid,
                    bbox=det["bbox"],
                    keypoints=det["keypoints"],
                    confidence=det["confidence"]
                )
                det["track_id"] = tid

        # Mark missed tracks and clean up aged tracks
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].time_since_update > self.max_age:
                    del self.tracks[tid]

        return detections
