import json
from pathlib import Path


SCHEMA_VERSION = "farpoint.display-names.v1"


def load_display_names(path):
    """Load local presentation names without changing artifact provenance."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if payload.get("schema_version") != SCHEMA_VERSION:
        return {}
    records = payload.get("records")
    if not isinstance(records, dict):
        return {}
    return {
        str(record_id): name.strip()
        for record_id, name in records.items()
        if isinstance(name, str) and name.strip()
    }
