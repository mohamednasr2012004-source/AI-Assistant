import numpy as np

METADATA_FIELDS = [
    "id",
    "source",
    "record_type",
    "title",
    "unit_type",
    "unit_type_group",
    "property_for",
    "price",
    "currency",
    "country",
    "city",
    "district",
    "completion_year",
    "language",
    "url",
]


def to_primitive(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
