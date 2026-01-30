import time
from typing import Dict


class PeopleAnalytics:
    def __init__(self):
        # track_id -> first_seen timestamp
        self.first_seen: Dict[int, float] = {}

        # track_id -> last_seen timestamp
        self.last_seen: Dict[int, float] = {}

    def update(self, tracks):
        """
        Update analytics state using current tracks.
        Only considers label == 'person'.
        """
        now = time.time()

        for tr in tracks:
            if tr["label"] != "person":
                continue

            tid = tr["track_id"]

            if tid not in self.first_seen:
                self.first_seen[tid] = now

            self.last_seen[tid] = now

    def current_count(self) -> int:
        """
        Number of people currently visible.
        """
        return len(self.last_seen)

    def unique_count(self) -> int:
        """
        Total unique people seen so far.
        """
        return len(self.first_seen)

    def dwell_times(self) -> Dict[int, float]:
        """
        Dwell time per person (seconds).
        """
        return {
            tid: self.last_seen[tid] - self.first_seen[tid]
            for tid in self.first_seen
            if tid in self.last_seen
        }

    def average_dwell_time(self) -> float:
        times = self.dwell_times().values()
        if not times:
            return 0.0
        return sum(times) / len(times)