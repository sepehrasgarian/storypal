"""Tier 3 storage: the curated data piles triage routes into.

Each record carries everything a training run would need: context,
the tutor's reply, the verdicts, and an empty corrected_reply slot an
offline curation step would fill to form SFT/DPO pairs.
"""

import json
from pathlib import Path

from api.config import CURATED_DIR
from api.triage import Route


class CuratedStore:
    def __init__(self, directory: str | Path = CURATED_DIR):
        self.directory = Path(directory)

    def add(self, route: Route, record: dict) -> None:
        if route is Route.ARCHIVE:
            return  # normal turns live in the trajectory log only
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {**record, "corrected_reply": None}
        with self._path(route).open("a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self, samples_per_pile: int = 3) -> dict:
        """Counts and latest samples for the UI's right panel."""
        piles = {}
        for route in (Route.REVIEW_QUEUE, Route.FINETUNE_SET):
            records = self._read(route)
            piles[route.value] = {"count": len(records), "samples": records[-samples_per_pile:]}
        return piles

    def _path(self, route: Route) -> Path:
        return self.directory / f"{route.value}.jsonl"

    def _read(self, route: Route) -> list[dict]:
        path = self._path(route)
        if not path.exists():
            return []
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
