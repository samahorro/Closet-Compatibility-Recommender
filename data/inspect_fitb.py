
import json
from pathlib import Path

path = (
    Path(__file__).resolve().parent
    / "raw"
    / "polyvore"
    / "disjoint"
    / "fill_in_blank_test.json"
)

with open(path, "r", encoding="utf-8") as f:
    records = json.load(f)

print("=== Fill-in-the-Blank Inspection ===")
print("Record count:", len(records))

if isinstance(records, list) and records:
    print("First record:")
    print(json.dumps(records[0], indent=2)[:3000])
elif isinstance(records, dict):
    print("Top-level keys:", list(records.keys())[:20])
    print(json.dumps(records, indent=2)[:3000])
