"""Convert INDIA_STATES.geojson (datta07/INDIAN-SHAPEFILES) into a compact
SVG-path JSON the studio can render inline.

- simplifies geometry by snapping to a ~1 km grid and dropping duplicates
- projects lon/lat into a 380x420 viewBox (equirectangular, lat-corrected)
- maps STNAME to Vaani config state names (None = state not in Vaani)

Run once:  .venv/bin/python scripts/build_map.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaanibox.districts import STATES

RAW = ROOT / "static" / "india_states_raw.geojson"
OUT = ROOT / "static" / "india_map.json"

VIEW_W, VIEW_H = 380, 420
SNAP = 0.02  # degrees, ~2 km — plenty for a 400px map

SPECIAL = {"JAMMU & KASHMIR": "JammuAndKashmir"}


def vaani_name(stname: str) -> str | None:
    if stname in SPECIAL:
        return SPECIAL[stname]
    candidate = "".join(w.capitalize() for w in stname.replace("&", "And").split())
    return candidate if candidate in STATES else None


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        yield poly[0]  # outer ring only; holes are invisible at this scale


def simplify(ring):
    out, prev = [], None
    for lon, lat in ring:
        pt = (round(lon / SNAP) * SNAP, round(lat / SNAP) * SNAP)
        if pt != prev:
            out.append(pt)
            prev = pt
    return out if len(out) >= 4 else []


RAW_URL = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_STATES.geojson"


def main():
    if not RAW.exists():
        import urllib.request

        print("downloading", RAW_URL)
        RAW.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW)
    d = json.load(open(RAW))
    features = d["features"]

    all_pts = [p for f in features for r in rings(f["geometry"]) for p in r]
    lons = [p[0] for p in all_pts]
    lats = [p[1] for p in all_pts]
    lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
    klon = math.cos(math.radians((lat0 + lat1) / 2))  # narrow the longitude scale
    scale = min(VIEW_W / ((lon1 - lon0) * klon), VIEW_H / (lat1 - lat0))

    def project(lon, lat):
        return round((lon - lon0) * klon * scale, 1), round((lat1 - lat) * scale, 1)

    states = []
    for f in features:
        stname = f["properties"]["STNAME"]
        parts = []
        for ring in rings(f["geometry"]):
            pts = [project(*p) for p in simplify(ring)]
            if not pts:
                continue
            path = f"M{pts[0][0]} {pts[0][1]}" + "".join(f"L{x} {y}" for x, y in pts[1:]) + "Z"
            parts.append(path)
        if parts:
            states.append(
                {"id": vaani_name(stname), "name": f["properties"]["STNAME_SH"], "d": "".join(parts)}
            )

    OUT.write_text(json.dumps({"viewBox": f"0 0 {VIEW_W} {VIEW_H}", "states": states}))
    mapped = sum(1 for s in states if s["id"])
    print(f"{len(states)} states written ({mapped} in Vaani), {OUT.stat().st_size / 1024:.0f} KB -> {OUT}")
    unmapped_vaani = set(STATES) - {s["id"] for s in states}
    print("Vaani states without a shape:", unmapped_vaani or "none")


if __name__ == "__main__":
    main()
