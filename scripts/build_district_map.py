"""Convert INDIA_DISTRICTS.geojson (datta07/INDIAN-SHAPEFILES, 74 MB) into a
compact SVG-path JSON for the studio's Accent Atlas.

- aggressive grid simplification (~3 km) — it's a 400px map
- lat-corrected equirectangular projection into a 380x420 viewBox
- district names matched to Vaani "State_District" configs: exact on a
  normalized key, then difflib fuzzy within the same state, then a small
  hand-curated alias table for the stubborn ones

Run once:  .venv/bin/python scripts/build_district_map.py
"""

import difflib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaanibox.districts import DISTRICT_CONFIGS

RAW = ROOT / "static" / "india_districts_raw.geojson"
OUT = ROOT / "static" / "india_districts_map.json"
RAW_URL = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"

VIEW_W, VIEW_H = 380, 420
SNAP = 0.03  # degrees

# geojson "state|district" (normalized) -> Vaani config, for names too far
# apart for fuzzy matching.
ALIASES = {
    "jammukashmir|srinagar": "JammuAndKashmir_Srinagar",
    "andhrapradesh|visakhapatnam": "AndhraPradesh_Vishakapattanam",
    "andhrapradesh|anantapur": "AndhraPradesh_Anantpur",
    "andhrapradesh|annamayya": "AndhraPradesh_Annamaya",
    "andhrapradesh|parvathipurammanyam": "AndhraPradesh_Manyam",
    "andhrapradesh|srisathyasai": "AndhraPradesh_SriSatyaSai",
    "bihar|jehanabad": "Bihar_Jahanabad",
    "bihar|purbichamparan": "Bihar_EastChamparan",
    "bihar|pashchimchamparan": "Bihar_WestChamparan",
    "chhattisgarh|kawardha": "Chhattisgarh_Kabirdham",
    "goa|northgoa": "Goa_NorthSouthGoa",
    "goa|southgoa": "Goa_NorthSouthGoa",
    "karnataka|bengaluruurban": "Karnataka_Bangalore",
    "karnataka|belagavi": "Karnataka_Belgaum",
    "karnataka|ballari": "Karnataka_Bellary",
    "karnataka|vijayapura": "Karnataka_Bijapur",
    "karnataka|kalaburagi": "Karnataka_Gulbarga",
    "karnataka|mysuru": "Karnataka_Mysore",
    "karnataka|shivamogga": "Karnataka_Shimoga",
    "maharashtra|chhatrapatisambhajinagar": "Maharashtra_Aurangabad",
    "meghalaya|southgarohills": "Meghalaya_SouthGarohills",
    "odisha|khordha": "Odisha_Khordha",
    "tamilnadu|kancheepuram": None,
    "telangana|kumurambheemasifabad": "Telangana_KomaramBheem",
    "uttarpradesh|amroha": "UttarPradesh_JyotibaPhuleNagar",
    "westbengal|purbamedinipur": None,
    "westbengal|paschimmedinipur": "WestBengal_PaschimMedinipur",
    "westbengal|dakshindinajpur": "WestBengal_DakshinDinajpur",
    "delhi|delhi": "Delhi_NewDelhi",
    # truncated-at-the-diacritic Karnataka names
    "karnataka|ball": "Karnataka_Bellary",
    "karnataka|belag": "Karnataka_Belgaum",
    "karnataka|ch": "Karnataka_Chamarajanagar",
    "karnataka|dh": "Karnataka_Dharwad",
    "karnataka|bengaaruurban": "Karnataka_Bangalore",
    "karnataka|bengaluruurban": "Karnataka_Bangalore",
    "karnataka|mysuru": "Karnataka_Mysore",
    "andhrapradesh|ananthapuramu": "AndhraPradesh_Anantpur",
    "assam|kamrupmetro": "Assam_KamrupMetropolitan",
    "bihar|pashchimichamparan": "Bihar_WestChamparan",
    "chhattisgarh|kawardhakabirdham": "Chhattisgarh_Kabirdham",
    "chhattisgarh|korea": "Chhattisgarh_Koriya",
    "odisha|nuaparha": "Odisha_Nuapada",
    "westbengal|kochbihar": "WestBengal_CoochBehar",
    "westbengal|darjiling": "WestBengal_Darjeeling",
    "westbengal|maldah": "WestBengal_Malda",
}


# The source file has mojibake: long-vowel diacritics became ASCII symbols
# (Mys#ru = Mysuru, KOCH BIH>R = Koch Bihar, B|RBH@M = Birbhum, B\dar = Bidar).
MOJIBAKE = str.maketrans({"#": "u", ">": "a", "|": "i", "@": "u", "\\": "i"})


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").translate(MOJIBAKE).lower())


# Vaani lookup: normalized state -> {normalized district: config}
VAANI: dict[str, dict[str, str]] = {}
for cfg in DISTRICT_CONFIGS:
    st, dt = cfg.split("_", 1)
    VAANI.setdefault(norm(st), {})[norm(dt)] = cfg


def match(state: str, district: str) -> str | None:
    ns, nd = norm(state), norm(district)
    alias_key = f"{ns}|{nd}"
    if alias_key in ALIASES:
        return ALIASES[alias_key]
    table = VAANI.get(ns)
    if not table:
        # geojson uses e.g. "JAMMU AND KASHMIR"; try loose state match
        hits = difflib.get_close_matches(ns, list(VAANI), n=1, cutoff=0.8)
        table = VAANI.get(hits[0]) if hits else None
    if not table:
        return None
    if nd in table:
        return table[nd]
    hits = difflib.get_close_matches(nd, list(table), n=1, cutoff=0.82)
    return table[hits[0]] if hits else None


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        yield poly[0]


def simplify(ring):
    out, prev = [], None
    for lon, lat in ring:
        pt = (round(lon / SNAP) * SNAP, round(lat / SNAP) * SNAP)
        if pt != prev:
            out.append(pt)
            prev = pt
    return out if len(out) >= 4 else []


def main():
    if not RAW.exists():
        import urllib.request

        print("downloading", RAW_URL)
        urllib.request.urlretrieve(RAW_URL, RAW)
    d = json.load(open(RAW))
    features = d["features"]

    all_pts = [p for f in features for r in rings(f["geometry"]) for p in r[:: max(1, len(r) // 50)]]
    lons = [p[0] for p in all_pts]
    lats = [p[1] for p in all_pts]
    lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
    klon = math.cos(math.radians((lat0 + lat1) / 2))
    scale = min(VIEW_W / ((lon1 - lon0) * klon), VIEW_H / (lat1 - lat0))

    def project(lon, lat):
        return round((lon - lon0) * klon * scale, 1), round((lat1 - lat) * scale, 1)

    out, matched = [], set()
    for f in features:
        props = f["properties"]
        state, district = props.get("state") or "", props.get("district") or ""
        cfg = match(state, district)
        parts = []
        for ring in rings(f["geometry"]):
            pts = [project(*p) for p in simplify(ring)]
            if len(pts) < 4:
                continue
            dedup, prev = [], None
            for p in pts:
                if p != prev:
                    dedup.append(p)
                    prev = p
            if len(dedup) < 4:
                continue
            parts.append(f"M{dedup[0][0]} {dedup[0][1]}" + "".join(f"L{x} {y}" for x, y in dedup[1:]) + "Z")
        if parts:
            display = (cfg.split("_", 1)[1] if cfg else district.translate(MOJIBAKE).title()) or "?"
            out.append({"id": cfg, "name": display, "st": state.title(), "d": "".join(parts)})
            if cfg:
                matched.add(cfg)

    OUT.write_text(json.dumps({"viewBox": f"0 0 {VIEW_W} {VIEW_H}", "states": out}))
    print(f"{len(out)} districts written, {OUT.stat().st_size / 1048576:.2f} MB -> {OUT.name}")
    missing = sorted(set(DISTRICT_CONFIGS) - matched)
    print(f"Vaani configs matched: {len(matched)}/{len(DISTRICT_CONFIGS)}")
    if missing:
        print("unmatched Vaani configs:", missing)


if __name__ == "__main__":
    main()
