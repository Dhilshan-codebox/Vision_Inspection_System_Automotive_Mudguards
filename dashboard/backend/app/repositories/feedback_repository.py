import json
from pathlib import Path
from typing import Dict, Any, List


class FeedbackRepository:
    def __init__(self, storage_path: str = "logs/review_queue.jsonl"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_feedback(self, feedback_data: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data) + "\n")

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        items = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items
