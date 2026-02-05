#!/usr/bin/env python3
"""
Convert debug-heavy spatial join output into a small "handoff" JSON.

Input:  per_coordinate_spatial_join.json  (output of spatial.py)
Output: per_coordinate_handoff.json       (minimal, team-friendly)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


INP = Path("per_coordinate_spatial_join.json")
OUT = Path("per_coordinate_handoff.json")

# If you want an even smaller file, set this False to drop URLs.
INCLUDE_IMAGE_URLS = True


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def building_display_name(tags: Dict[str, Any]) -> Optional[str]:
    return tags.get("name") or tags.get("short_name")


def slim_image_entry(img: Dict[str, Any]) -> Dict[str, Any]:
    pose = img.get("pose") or {}
    best = img.get("best_candidate") or {}

    out = {
        "image_id": img.get("image_id"),
        "captured_at": pose.get("captured_at"),
        "pose": {
            "lat": pose.get("lat"),
            "lon": pose.get("lon"),
            "heading": pose.get("heading"),
        },
        # facade / plausibility fields (useful downstream)
        "closest_edge_index": best.get("closest_edge_index"),
        "distance_to_edge_m": best.get("distance_to_edge_m"),
        "heading_delta_deg": best.get("heading_delta_deg"),
        "visibility_score": best.get("visibility_score"),
    }

    if INCLUDE_IMAGE_URLS:
        out["url"] = img.get("url")

    return out


def slim_building_entry(b: Dict[str, Any]) -> Dict[str, Any]:
    tags = b.get("tags") or {}

    # Keep tags minimal: name + a couple optional hints.
    minimal_tags = {}
    for k in ["name", "short_name", "building", "amenity", "building:levels", "height", "roof:shape"]:
        if k in tags:
            minimal_tags[k] = tags[k]

    assigned_images = b.get("assigned_images") or []
    assigned_images_slim = [slim_image_entry(img) for img in assigned_images]

    # Optional: if you want to preserve your building summary (already compact), keep it.
    summary = b.get("summary")

    out = {
        "osm_id": b.get("osm_id"),
        "name": building_display_name(tags),
        "tags": minimal_tags,
        "height_m": b.get("height_m"),
        "centroid_latlon": b.get("centroid_latlon"),
        "footprint_latlon": b.get("footprint_latlon"),
        "assigned_images": assigned_images_slim,
    }

    if isinstance(summary, dict):
        out["summary"] = summary

    return out


def slim_record(record: Dict[str, Any]) -> Dict[str, Any]:
    coord = record.get("coordinate") or {}
    bbox = record.get("bbox_south_west_north_east")

    buildings = record.get("buildings_joined") or []
    buildings_slim = [slim_building_entry(b) for b in buildings]

    # Only keep images that actually passed and were assigned
    # (in your debug output, rejected images still show up in mapillary_joined)
    map_joined = record.get("mapillary_joined") or []
    assigned_images = []
    for img in map_joined:
        if img.get("assigned_building") is None:
            continue
        assigned_images.append({
            "image_id": img.get("image_id"),
            "assigned_building": img.get("assigned_building"),
            "closest_edge_index": safe_get(img, ["best_candidate", "closest_edge_index"]),
            "visibility_score": safe_get(img, ["best_candidate", "visibility_score"]),
        })

    out = {
        "coordinate": {"lat": coord.get("lat"), "lon": coord.get("lon")},
        "bbox_south_west_north_east": bbox,
        "buildings": buildings_slim,
        # Small index for quick lookups (optional but handy)
        "assigned_image_index": assigned_images,
        # Carry params (optional)
        "spatial_join_params": record.get("spatial_join_params"),
    }
    return out


def main():
    data = json.loads(INP.read_text())
    out = [slim_record(r) for r in data]
    OUT.write_text(json.dumps(out, indent=2))
    print(">>> wrote", OUT.resolve())


if __name__ == "__main__":
    main()
