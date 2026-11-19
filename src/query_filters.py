import difflib
import json
from typing import Any, Callable

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ValidationError

from src.prompt_catalog import get_filter_extraction_prompt
from utils.json_utils import parse_json_object
from utils.llm import get_chat_model
from utils.logger import get_logger

logger = get_logger(__name__)

EXCLUDED_FIELDS = {"language"}

CATALOG_FIELDS = [
    "city",
    "district",
    "country",
    "source",
    "property_for",
    "unit_type",
    "record_type",
]

CITY_ALIASES = {
    "جدة": "Jeddah",
    "الرياض": "Riyadh",
    "مكة": "Makkah Al Mukarramah",
    "مكة المكرمة": "Makkah Al Mukarramah",
    "المدينة": "Madinah",
    "المدينة المنورة": "Madinah",
    "الدمام": "Dammam",
    "الخبر": "Khobar",
    "الطائف": "Taif",
    "جازان": "Jazan",
    "تبوك": "Tabuk",
    "أبها": "Abha",
    "بريدة": "Buraydah",
    "حائل": "Hail",
    "نجران": "Najran",
}


class PropertyFilters(BaseModel):
    city: str | None = Field(
        None, description="City name exactly as it appears in the city catalog, or null."
    )
    district: str | None = Field(
        None, description="District/neighborhood exactly as it appears in the district catalog, or null."
    )
    country: str | None = Field(
        None, description="Country exactly as it appears in the country catalog, or null."
    )
    source: str | None = Field(
        None, description="Data source (wasalt.sa or darglobal.co.uk), or null."
    )
    property_for: str | None = Field(
        None, description="'sale' or 'rent' when the user asks to buy/rent, otherwise null."
    )
    unit_type: str | None = Field(
        None, description="Property/unit type exactly as it appears in the unit type catalog, or null."
    )
    record_type: str | None = Field(
        None, description="'listing' or 'project' when the user clearly asks for one, otherwise null."
    )
    project: str | None = Field(
        None, description="Project name exactly as it appears in the project catalog, or null."
    )
    price_min: float | None = Field(
        None, description="Minimum price in the listing currency, if the user gives a lower bound."
    )
    price_max: float | None = Field(
        None, description="Maximum price in the listing currency, if the user gives an upper bound."
    )
    completion_year: int | None = Field(
        None, description="Completion/delivery year, if the user asks for a specific year."
    )


def build_catalog(data: pd.DataFrame) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}

    for field in CATALOG_FIELDS:
        if field in data.columns:
            values = data[field].dropna().astype(str).str.strip()
            catalog[field] = sorted({value for value in values if value})
        else:
            catalog[field] = []

    if {"record_type", "title"}.issubset(data.columns):
        projects = data[data["record_type"] == "project"]["title"].dropna().astype(str)
        catalog["project"] = sorted({title.strip() for title in projects if title.strip()})
    else:
        catalog["project"] = []

    return catalog


class MetadataFilterExtractor:
    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str | None = None,
    ):
        self.catalog = build_catalog(data)
        self.chain = get_filter_extraction_prompt() | get_chat_model() | StrOutputParser()

    def _catalog_text(self) -> str:
        lines = []
        for field, values in self.catalog.items():
            if not values:
                continue
            lines.append(f"{field}: {', '.join(values)}")
        return "\n".join(lines)

    def extract(self, query: str) -> PropertyFilters:
        try:
            raw = self.chain.invoke(
                {"catalog": self._catalog_text(), "question": query}
            )
            payload = parse_json_object(raw)
            return PropertyFilters.model_validate(payload)
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            logger.warning("Could not extract metadata filters: %s", error)
            return PropertyFilters()
        except Exception as error:
            logger.warning("Metadata filter extraction failed: %s", error)
            return PropertyFilters()

    def _resolve(self, field: str, value: str | None) -> str | None:
        if not value:
            return None

        value = value.strip()
        if field == "city":
            value = CITY_ALIASES.get(value, value)
        if not value:
            return None

        options = self.catalog.get(field, [])
        if not options:
            return None

        lowered = {option.lower(): option for option in options}
        if value.lower() in lowered:
            return lowered[value.lower()]

        cutoff = 0.6 if field == "project" else 0.75
        match = difflib.get_close_matches(value, options, n=1, cutoff=cutoff)
        return match[0] if match else None

    def build_where(self, filters: PropertyFilters) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []

        for field in CATALOG_FIELDS:
            if field in EXCLUDED_FIELDS:
                continue
            resolved = self._resolve(field, getattr(filters, field, None))
            if resolved is not None and resolved != "":
                conditions.append({field: resolved})

        project = self._resolve("project", filters.project)
        if project is not None:
            conditions.append({"title": project})

        if filters.price_min is not None:
            conditions.append({"price": {"$gte": float(filters.price_min)}})
        if filters.price_max is not None:
            conditions.append({"price": {"$lte": float(filters.price_max)}})
        if filters.completion_year is not None:
            conditions.append({"completion_year": {"$eq": int(filters.completion_year)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def build_predicate(where: dict[str, Any] | None) -> Callable[[dict], bool] | None:
        if not where:
            return None
        return lambda metadata: matches_metadata(metadata, where)


def matches_metadata(metadata: dict, where: dict[str, Any]) -> bool:
    for key, condition in where.items():
        if key == "$and":
            if not all(matches_metadata(metadata, item) for item in condition):
                return False
            continue
        if key == "$or":
            if not any(matches_metadata(metadata, item) for item in condition):
                return False
            continue

        actual = metadata.get(key)
        if not isinstance(condition, dict):
            if actual != condition:
                return False
            continue

        for operator, expected in condition.items():
            if operator == "$eq" and actual != expected:
                return False
            if operator == "$ne" and actual == expected:
                return False
            if operator == "$in" and actual not in expected:
                return False
            if operator == "$nin" and actual in expected:
                return False
            if operator in {"$gte", "$lte", "$gt", "$lt"}:
                if actual is None:
                    return False
                if operator == "$gte" and not actual >= expected:
                    return False
                if operator == "$lte" and not actual <= expected:
                    return False
                if operator == "$gt" and not actual > expected:
                    return False
                if operator == "$lt" and not actual < expected:
                    return False

    return True
