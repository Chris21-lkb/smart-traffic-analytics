from typing import Dict, Any

# Global shared state (simple & explicit)
STATE: Dict[str, Any] = {
    "running": False,
    "people": {
        "current": 0,
        "unique": 0,
        "avg_dwell": 0.0,
    },
    "vehicles": {
        "current": 0,
        "per_class": {},
        "congestion": "UNKNOWN",
    },
}