import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"text", "source", "url"}


class RealEstateDataLoader:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def _read_records(self) -> list[dict]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        if self.file_path.suffix.lower() == ".jsonl":
            with self.file_path.open("r", encoding="utf-8") as file:
                return [json.loads(line) for line in file if line.strip()]

        if self.file_path.suffix.lower() == ".json":
            with self.file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                raise ValueError("JSON data must be a list of records or a single record object")
            return payload

        raise ValueError("Only .jsonl and .json files are supported")

    def load_data(self) -> pd.DataFrame:
        records = []

        for index, record in enumerate(self._read_records(), start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Record {index} must be an object")

            text = record.get("text")
            metadata = record.get("metadata") or {}

            if not isinstance(metadata, dict):
                raise ValueError(f"Record {index}: metadata must be an object")

            if not text or not str(text).strip():
                continue

            row = {"text": str(text).strip()}
            row.update(metadata)
            row["id"] = record.get("id") or metadata.get("url") or f"doc-{index}"
            records.append(row)

        if not records:
            raise ValueError("No usable records found in the data file")

        data = pd.DataFrame(records)
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        return data
