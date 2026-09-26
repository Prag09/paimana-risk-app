"""Sanity-check the India states/UTs GeoJSON the Risk Map uses.

Loads the exact file and property key components/charts.py loads (so this
script can never silently drift from what the app actually renders), and
checks that a handful of known reference points fall inside the state/UT
they belong to - in particular the disputed-territory points that decide
whether the map follows India's official claimed boundary.

Usage: python scripts/check_map.py [path/to/file.geojson]
Exits 0 if every check passes, 1 otherwise.
"""

import json
import sys
from pathlib import Path

from shapely.geometry import Point, shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.charts import GEOJSON_STATE_KEY, INDIA_GEOJSON_PATH  # noqa: E402

CHECKS = [
    ("Gilgit", 35.92, 74.31, {"Jammu & Kashmir", "Ladakh"}),
    ("Aksai Chin", 35.0, 79.0, {"Ladakh"}),
    ("Port Blair", 11.62, 92.73, {"Andaman & Nicobar"}),
    ("Kavaratti", 10.57, 72.64, {"Lakshadweep"}),
]


def load_shapes(geojson_path):
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)
    return {feat["properties"][GEOJSON_STATE_KEY]: shape(feat["geometry"]) for feat in gj["features"]}


def main():
    geojson_path = sys.argv[1] if len(sys.argv) > 1 else INDIA_GEOJSON_PATH
    print(f"Checking: {geojson_path}\n")

    shapes = load_shapes(geojson_path)
    all_passed = True

    for label, lat, lon, expected_states in CHECKS:
        point = Point(lon, lat)  # shapely is (x, y) = (lon, lat)
        matches = {name for name in expected_states if name in shapes and shapes[name].contains(point)}
        passed = bool(matches)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        detail = f"inside {sorted(matches)}" if matches else f"NOT inside any of {sorted(expected_states)}"
        print(f"[{status}] {label} ({lat}, {lon}) expected in {sorted(expected_states)} - {detail}")

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
