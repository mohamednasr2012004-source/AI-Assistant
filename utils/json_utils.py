import json
import re
from typing import Any


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model JSON response is not an object")
    return payload
