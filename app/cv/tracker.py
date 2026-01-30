from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boxes in [x1,y1,x2,y2]"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    label: str
    bbox: np.ndarray  # xyxy
    hits: int = 1
    time_since_update: int = 0


class IoUTracker:
    """
    Simple multi-object tracker using IoU matching.
    - class-aware matching (person matches person, car matches car, etc.)
    - assigns stable IDs as long as IoU stays decent
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _new_track(self, label: str, bbox: np.ndarray) -> Track:
        tid = self._next_id
        self._next_id += 1
        tr = Track(track_id=tid, label=label, bbox=bbox.copy())
        self.tracks[tid] = tr
        return tr

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: list of {"bbox": np.array/int list xyxy, "label": str, "score": float}
        returns: list of dicts with track info:
                 {"track_id", "label", "bbox", "score"}
        """

        # Age all tracks (assume they were not updated yet)
        for tr in self.tracks.values():
            tr.time_since_update += 1

        # Group detections by label
        dets_by_label: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        for d in detections:
            bbox = np.array(d["bbox"], dtype=int)
            dets_by_label.setdefault(d["label"], []).append((bbox, float(d["score"])))

        # Match per label (class-aware)
        for label, det_list in dets_by_label.items():
            # Existing tracks for this label
            track_ids = [tid for tid, tr in self.tracks.items() if tr.label == label]
            if not track_ids:
                for bbox, _score in det_list:
                    self._new_track(label, bbox)
                continue

            track_boxes = np.array([self.tracks[tid].bbox for tid in track_ids], dtype=float)
            det_boxes = np.array([b for b, _s in det_list], dtype=float)

            # IoU matrix: tracks x detections
            iou_mat = np.zeros((len(track_ids), len(det_list)), dtype=float)
            for i in range(len(track_ids)):
                for j in range(len(det_list)):
                    iou_mat[i, j] = iou_xyxy(track_boxes[i], det_boxes[j])

            # Greedy matching (simple, works well enough to start)
            matched_tracks = set()
            matched_dets = set()

            while True:
                if iou_mat.size == 0:
                    break
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best = iou_mat[i, j]
                if best < self.iou_threshold:
                    break

                tid = track_ids[i]
                if tid in matched_tracks or j in matched_dets:
                    iou_mat[i, j] = -1
                    continue

                # Assign detection j to track tid
                bbox_j, _score = det_list[j]
                tr = self.tracks[tid]
                tr.bbox = bbox_j.copy()
                tr.hits += 1
                tr.time_since_update = 0

                matched_tracks.add(tid)
                matched_dets.add(j)

                # Prevent reusing this row/col
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1

            # Unmatched detections → new tracks
            for j in range(len(det_list)):
                if j not in matched_dets:
                    bbox_j, _score = det_list[j]
                    self._new_track(label, bbox_j)

        # Remove dead tracks
        dead = [tid for tid, tr in self.tracks.items() if tr.time_since_update > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        # Build output track list
        outputs = []
        for tid, tr in self.tracks.items():
            # Only output tracks that were seen recently (optional)
            if tr.time_since_update <= 2:
                outputs.append({
                    "track_id": tid,
                    "label": tr.label,
                    "bbox": tr.bbox.astype(int),
                    "hits": tr.hits
                })

        return outputs