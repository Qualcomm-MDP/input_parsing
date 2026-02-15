#!/usr/bin/env python3
print(">>> LOADED:", __file__)

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# -----------------------------
# 1. Geo Math Helpers
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def wrap_angle_deg(a: float) -> float:
    return (a + 360) % 360

def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180) % 360 - 180
    return abs(d)

def meters_per_degree(lat: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * math.cos(lat_rad)
    return m_per_deg_lat, m_per_deg_lon

def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    mlat, mlon = meters_per_degree(lat0)
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x, y

def point_to_segment_distance_m(p_lat, p_lon, a_lat, a_lon, b_lat, b_lon) -> Tuple[float, float]:
    lat0, lon0 = p_lat, p_lon
    px, py = latlon_to_local_xy(p_lat, p_lon, lat0, lon0)
    ax, ay = latlon_to_local_xy(a_lat, a_lon, lat0, lon0)
    bx, by = latlon_to_local_xy(b_lat, b_lon, lat0, lon0)
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy), t

def clean_polygon_latlon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not poly: return poly
    cleaned = [poly[0]]
    for p in poly[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    if len(cleaned) >= 3:
        if cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0])
    return cleaned

def polygon_area_m2(polygon_latlon: List[Tuple[float, float]]) -> float:
    if len(polygon_latlon) < 3: return 0.0
    poly = polygon_latlon
    if len(poly) >= 4 and poly[0] == poly[-1]: poly = poly[:-1]
    if len(poly) < 3: return 0.0
    lat0 = sum(p[0] for p in poly) / len(poly)
    lon0 = sum(p[1] for p in poly) / len(poly)
    pts = [latlon_to_local_xy(lat, lon, lat0, lon0) for lat, lon in poly]
    area2 = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5

@dataclass
class EdgeInfo:
    distance_m: float
    edge_index: int
    closest_point_lat: float
    closest_point_lon: float
    edge_bearing_deg: float
    facade_normal_deg: float

def nearest_edge_info(cam_lat, cam_lon, polygon_latlon_closed) -> Optional[EdgeInfo]:
    if len(polygon_latlon_closed) < 4: return None
    best_d = float("inf")
    best_i = -1
    best_t = 0.0
    for i in range(len(polygon_latlon_closed) - 1):
        a_lat, a_lon = polygon_latlon_closed[i]
        b_lat, b_lon = polygon_latlon_closed[i + 1]
        d, t = point_to_segment_distance_m(cam_lat, cam_lon, a_lat, a_lon, b_lat, b_lon)
        if d < best_d:
            best_d, best_i, best_t = d, i, t

    if best_i == -1: return None

    a_lat, a_lon = polygon_latlon_closed[best_i]
    b_lat, b_lon = polygon_latlon_closed[best_i + 1]
    cp_lat = a_lat + best_t * (b_lat - a_lat)
    cp_lon = a_lon + best_t * (b_lon - a_lon)
    e_bearing = bearing_deg(a_lat, a_lon, b_lat, b_lon)
    n1 = wrap_angle_deg(e_bearing + 90)
    n2 = wrap_angle_deg(e_bearing - 90)
    b_cp_to_cam = bearing_deg(cp_lat, cp_lon, cam_lat, cam_lon)
    n = n1 if angle_diff_deg(n1, b_cp_to_cam) <= angle_diff_deg(n2, b_cp_to_cam) else n2

    return EdgeInfo(best_d, best_i, cp_lat, cp_lon, e_bearing, n)

def parse_height_m(tags: Dict[str, Any]) -> Optional[float]:
    h = tags.get("height")
    if isinstance(h, str):
        try:
            s = h.strip().lower().replace("meters", "m").replace(" ", "")
            if s.endswith("m"): s = s[:-1]
            return float(s)
        except: pass
    return None

# -----------------------------
# 2. OSM Extraction (Robust: Ways & Relations)
# -----------------------------

def _way_nodes_latlon(way, nodes_by_id):
    return [nodes_by_id.get(nid) for nid in (way.get("nodes") or []) if nodes_by_id.get(nid)]

def _extract_relation_multipolygon_rings(rel, ways_by_id, nodes_by_id):
    # Simplified ring stitcher for relations
    outer_ways = []
    for m in (rel.get("members") or []):
        if m.get("type") == "way" and m.get("role") in ["outer", ""]:
            w = ways_by_id.get(m.get("ref"))
            if w: outer_ways.append(_way_nodes_latlon(w, nodes_by_id))
    
    # Just take the largest closed loop found (simplified for robustness)
    best_poly = []
    for w in outer_ways:
        if len(w) > len(best_poly): best_poly = w
    
    if len(best_poly) > 2:
        return [clean_polygon_latlon(best_poly)], []
    return [], []

def extract_osm_buildings(overpass_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    elements = overpass_json.get("elements", []) if isinstance(overpass_json, dict) else []
    nodes_by_id = {e["id"]: (e["lat"], e["lon"]) for e in elements if e.get("type") == "node"}
    ways_by_id = {e["id"]: e for e in elements if e.get("type") == "way"}
    buildings = []

    # 1. Ways
    for e in elements:
        if e.get("type") != "way": continue
        tags = e.get("tags", {}) or {}
        if "building" not in tags: continue
        footprint = clean_polygon_latlon(_way_nodes_latlon(e, nodes_by_id))
        if len(footprint) < 4: continue
        buildings.append({
            "osm_id": f"way/{e.get('id')}",
            "tags": tags,
            "height_m": parse_height_m(tags),
            "footprint_latlon": footprint,
            "assigned_images": [] # Prepare for assignment
        })

    # 2. Relations
    for e in elements:
        if e.get("type") != "relation": continue
        tags = e.get("tags", {}) or {}
        if "building" not in tags: continue
        outer_rings, _ = _extract_relation_multipolygon_rings(e, ways_by_id, nodes_by_id)
        if not outer_rings: continue
        primary = outer_rings[0]
        if len(primary) < 4: continue
        buildings.append({
            "osm_id": f"relation/{e.get('id')}",
            "tags": tags,
            "height_m": parse_height_m(tags),
            "footprint_latlon": primary,
            "assigned_images": [] # Prepare for assignment
        })

    return buildings

# -----------------------------
# 3. Join Logic (Assign Everything)
# -----------------------------

def mapillary_pose(img: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    coords = img.get("computed_geometry", {}).get("coordinates", None)
    if not isinstance(coords, list) or len(coords) < 2: return None
    lon, lat = coords[0], coords[1]
    heading = img.get("computed_compass_angle", 0.0)
    return {"lat": float(lat), "lon": float(lon), "heading": float(heading), "captured_at": img.get("captured_at")}

def spatial_join_one_coordinate(record: Dict[str, Any]) -> Dict[str, Any]:
    osm_raw = record.get("osm_raw") or record.get("osm")
    mapillary = record.get("mapillary", []) or []
    
    # Extract buildings (using robust method)
    buildings = extract_osm_buildings(osm_raw) if osm_raw else []
    
    # Create a map to easily add images to buildings later
    b_by_id = {b["osm_id"]: b for b in buildings}
    
    kept_images = []

    print(f">> Processing batch: {len(mapillary)} images, {len(buildings)} buildings.")

    for img in mapillary:
        pose = mapillary_pose(img)
        url = img.get("thumb_original_url")
        image_id = img.get("id")

        if not url or not pose:
            continue # Can't assign if no data

        cam_lat, cam_lon, heading = pose["lat"], pose["lon"], pose["heading"]

        # -----------------------------
        # FIND CANDIDATES
        # -----------------------------
        candidates = []
        for b in buildings:
            outer = b["footprint_latlon"]
            ei = nearest_edge_info(cam_lat, cam_lon, outer)
            
            if ei is None: continue

            # Calculate geometry details
            b_cam_to_cp = bearing_deg(cam_lat, cam_lon, ei.closest_point_lat, ei.closest_point_lon)
            heading_delta = angle_diff_deg(heading, b_cam_to_cp)
            
            # Simple Scoring: Prefer closer buildings, then better angles
            # Lower score is better here for sorting
            sort_score = ei.distance_m
            
            candidates.append({
                "osm_id": b["osm_id"],
                "distance_to_edge_m": round(ei.distance_m, 3),
                "closest_edge_index": ei.edge_index,
                "heading_delta_deg": round(heading_delta, 2),
                "sort_score": sort_score
            })

        # -----------------------------
        # ASSIGN BEST (NO FILTERING)
        # -----------------------------
        
        # Sort by distance (closest first)
        candidates.sort(key=lambda c: c["sort_score"])
        
        best = candidates[0] if candidates else None
        
        assigned_id = best["osm_id"] if best else None

        # Build the exact object structure you liked
        join_info = {
            "image_id": image_id,
            "url": url,
            "pose": pose,
            "best_candidate": best,
            "candidates": candidates[:5], # Keep top 5 for reference
            "assigned_building": assigned_id, 
            "local_thumb_path": None, # Requested: No download
            "reason": "forced_assignment"
        }

        # Add to the building's list
        if assigned_id and assigned_id in b_by_id:
            b_by_id[assigned_id]["assigned_images"].append(join_info)
        
        # Add to the main list
        kept_images.append(join_info)

    # Prepare final record output
    record_out = dict(record)
    
    # Clear raw mapillary to save space
    if "mapillary" in record_out: del record_out["mapillary"]
    
    # Add our processed lists
    record_out["buildings_joined"] = list(b_by_id.values())
    record_out["mapillary_joined"] = kept_images
    
    return record_out


def main():
    inp = Path("per_coordinate_osm_mapillary.json")
    out = Path("per_coordinate_spatial_join_all.json")

    if not inp.exists():
        print(f"xx Error: {inp} not found.")
        return

    print(">>> reading", inp.resolve())
    data = json.loads(inp.read_text())

    # Process all
    joined = [spatial_join_one_coordinate(r) for r in data]

    out.write_text(json.dumps(joined, indent=2))
    print(">>> wrote", out.resolve())


if __name__ == "__main__":
    main()