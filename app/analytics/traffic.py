from collections import defaultdict
from typing import Dict, List


class VehicleAnalytics:
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self):
        # track_id -> label
        self.active_tracks: Dict[int, str] = {}

        # cumulative unique counts per class
        self.unique_counts = defaultdict(int)

    def update(self, tracks: List[Dict]):
        """
        Update vehicle analytics using current tracked objects.
        """
        current_ids = set()

        for tr in tracks:
            label = tr["label"]
            if label not in self.VEHICLE_CLASSES:
                continue

            tid = tr["track_id"]
            current_ids.add(tid)

            if tid not in self.active_tracks:
                self.active_tracks[tid] = label
                self.unique_counts[label] += 1

        # Remove inactive tracks
        self.active_tracks = {
            tid: lbl for tid, lbl in self.active_tracks.items()
            if tid in current_ids
        }

    def current_count(self) -> int:
        return len(self.active_tracks)

    def current_counts_per_class(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for lbl in self.active_tracks.values():
            counts[lbl] += 1
        return dict(counts)

    def congestion_level(self) -> str:
        """
        Very simple congestion proxy based on number of vehicles.
        """
        n = self.current_count()
        if n < 5:
            return "LOW"
        elif n < 15:
            return "MEDIUM"
        else:
            return "HIGH"