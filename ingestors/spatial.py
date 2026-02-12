#!/usr/bin/env python3
print(">>> LOADED:", __file__)

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# -----------------------------
# Small geo helpers
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Bearing from (lat1,lon1) to (lat2,lon2). 0=N, 90=E."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def wrap_angle_deg(a: float) -> float:
    return (a + 360) % 360


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between angles."""
    d = (a - b + 180) % 360 - 180
    return abs(d)


def meters_per_degree(lat: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * math.cos(lat_rad)
    return m_per_deg_lat, m_per_deg_lon


def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Local equirectangular-ish projection around (lat0, lon0)."""
    mlat, mlon = meters_per_degree(lat0)
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x, y


def local_xy_to_latlon(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    mlat, mlon = meters_per_degree(lat0)
    lat = lat0 + (y / mlat)
    lon = lon0 + (x / mlon)
    return lat, lon


def polygon_area_m2(polygon_latlon: List[Tuple[float, float]]) -> float:
    """Approx polygon area (m^2) in local coords around polygon centroid."""
    if len(polygon_latlon) < 3:
        return 0.0
    lat0 = sum(p[0] for p in polygon_latlon) / len(polygon_latlon)
    lon0 = sum(p[1] for p in polygon_latlon) / len(polygon_latlon)
    pts = [latlon_to_local_xy(lat, lon, lat0, lon0) for lat, lon in polygon_latlon]
    area2 = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def clean_polygon_latlon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Remove consecutive duplicates + ensure closure."""
    if not poly:
        return poly
    cleaned = [poly[0]]
    for p in poly[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    # Ensure at least 3 vertices (not counting closure)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    if len(cleaned) >= 3:
        # Close
        if cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0])
    return cleaned


def point_in_polygon(lat: float, lon: float, polygon_latlon: List[Tuple[float, float]]) -> bool:
    """Ray casting. polygon_latlon must be closed or will still work mostly."""
    x, y = lon, lat
    inside = False
    n = len(polygon_latlon)
    if n < 3:
        return False
    for i in range(n - 1):
        y1, x1 = polygon_latlon[i]
        y2, x2 = polygon_latlon[i + 1]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def is_approx_daylight(lat: float, lon: float, captured_at_ms: Optional[int]) -> bool:
    """
    Checks if photo was likely taken during the day.
    Uses approx local solar time (UTC + lon/15).
    Rejects if before 7am or after 7pm local solar time.
    """
    if captured_at_ms is None:
        return True # Default to keep if no time data

    try:
        # Mapillary captured_at is usually milliseconds epoch
        dt_utc = datetime.fromtimestamp(captured_at_ms / 1000.0, tz=timezone.utc)
        
        # Approx offset in hours (15 degrees = 1 hour)
        hour_offset = lon / 15.0
        local_hour = (dt_utc.hour + (dt_utc.minute / 60.0) + hour_offset) % 24
        
        # Strict daylight window: 7:00 AM to 7:00 PM
        if 7.0 <= local_hour <= 19.0:
            return True
        return False
    except Exception:
        return True # Fallback if parsing fails


def point_to_segment_distance_m(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float
) -> Tuple[float, float]:
    """Returns (distance_m, t) where t is the clamped projection along AB in local meters."""
    lat0 = p_lat
    lon0 = p_lon
    px, py = latlon_to_local_xy(p_lat, p_lon, lat0, lon0)
    ax, ay = latlon_to_local_xy(a_lat, a_lon, lat0, lon0)
    bx, by = latlon_to_local_xy(b_lat, b_lon, lat0, lon0)

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby

    if ab2 <= 1e-12:
        # Degenerate edge: treat as point
        return math.hypot(px - ax, py - ay), 0.0

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy), t


@dataclass
class EdgeInfo:
    distance_m: float
    edge_index: int
    closest_point_lat: float
    closest_point_lon: float
    edge_bearing_deg: float
    facade_normal_deg: float  # chosen to point roughly toward camera


def nearest_edge_info(
    cam_lat: float, cam_lon: float,
    polygon_latlon_closed: List[Tuple[float, float]]
) -> Optional[EdgeInfo]:
    """Find nearest polygon edge to camera point; returns edge index, closest point, and bearings."""
    if len(polygon_latlon_closed) < 4:
        return None

    best_d = float("inf")
    best_i = -1
    best_t = 0.0

    # polygon is closed: last == first
    for i in range(len(polygon_latlon_closed) - 1):
        a_lat, a_lon = polygon_latlon_closed[i]
        b_lat, b_lon = polygon_latlon_closed[i + 1]
        d, t = point_to_segment_distance_m(cam_lat, cam_lon, a_lat, a_lon, b_lat, b_lon)
        if d < best_d:
            best_d = d
            best_i = i
            best_t = t

    a_lat, a_lon = polygon_latlon_closed[best_i]
    b_lat, b_lon = polygon_latlon_closed[best_i + 1]

    cp_lat = a_lat + best_t * (b_lat - a_lat)
    cp_lon = a_lon + best_t * (b_lon - a_lon)

    # Edge bearing along A->B
    e_bearing = bearing_deg(a_lat, a_lon, b_lat, b_lon)

    # Pick facade normal that points toward the camera (approx)
    # Two candidates: edge bearing +/- 90
    n1 = wrap_angle_deg(e_bearing + 90)
    n2 = wrap_angle_deg(e_bearing - 90)
    # Choose the one closer to bearing from closest-point -> camera
    b_cp_to_cam = bearing_deg(cp_lat, cp_lon, cam_lat, cam_lon)
    n = n1 if angle_diff_deg(n1, b_cp_to_cam) <= angle_diff_deg(n2, b_cp_to_cam) else n2

    return EdgeInfo(
        distance_m=best_d,
        edge_index=best_i,
        closest_point_lat=cp_lat,
        closest_point_lon=cp_lon,
        edge_bearing_deg=e_bearing,
        facade_normal_deg=n,
    )


def dynamic_heading_threshold_deg(distance_m: float) -> float:
    """Farther away => stricter heading requirement."""
    if distance_m <= 15:
        return 60.0
    if distance_m <= 30:
        return 45.0
    if distance_m <= 50:
        return 30.0
    return 25.0


def parse_height_m(tags: Dict[str, Any]) -> Optional[float]:
    """Best-effort parse OSM height/building:levels."""
    # 1) explicit height
    h = tags.get("height")
    if isinstance(h, str):
        s = h.strip().lower().replace("meters", "m").replace(" ", "")
        # Accept "12", "12m", "12.5m"
        try:
            if s.endswith("m"):
                s = s[:-1]
            return float(s)
        except Exception:
            pass

    # 2) building:levels
    lv = tags.get("building:levels")
    if isinstance(lv, str):
        try:
            levels = float(lv.strip())
            # 3m/level is a common heuristic (can tune by region)
            return max(1.0, levels) * 3.0
        except Exception:
            pass

    return None


# -----------------------------
# OSM extraction (ways-only)
# -----------------------------

def extract_osm_buildings(overpass_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    elements = overpass_json.get("elements", [])
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in elements if e.get("type") == "node"}

    buildings = []
    for e in elements:
        if e.get("type") != "way":
            continue
        tags = e.get("tags", {}) or {}
        if "building" not in tags:
            continue

        footprint = [nodes.get(nid) for nid in (e.get("nodes") or [])]
        footprint = [p for p in footprint if p and p[0] is not None and p[1] is not None]
        if len(footprint) < 3:
            continue

        footprint = clean_polygon_latlon(footprint)
        if len(footprint) < 4:
            continue

        # Skip tiny/degenerate polygons
        if polygon_area_m2(footprint[:-1]) < 5.0:
            continue

        # centroid (simple avg of vertices, ok for our purposes)
        verts = footprint[:-1]
        c_lat = sum(p[0] for p in verts) / len(verts)
        c_lon = sum(p[1] for p in verts) / len(verts)

        buildings.append({
            "osm_id": e.get("id"),
            "tags": tags,
            "height_m": parse_height_m(tags),
            "footprint_latlon": footprint,  # closed
            "centroid_latlon": (c_lat, c_lon),
        })

    return buildings


# -----------------------------
# Join logic (top-K candidates + gating + thinning)
# -----------------------------

def mapillary_pose(img: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    coords = img.get("computed_geometry", {}).get("coordinates", None)
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    lon, lat = coords[0], coords[1]
    heading = img.get("computed_compass_angle", None)
    if lat is None or lon is None or heading is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "heading": float(heading),
        "captured_at": img.get("captured_at"),
    }


def score_candidate(distance_m: float, heading_delta_deg: float) -> float:
    """Simple scoring to rank candidates."""
    score = 0.0
    # Note: 'inside' is no longer scored here because we filter it out earlier

    # distance contribution
    if distance_m <= 15:
        score += 2.0
    elif distance_m <= 30:
        score += 1.5
    elif distance_m <= 50:
        score += 1.0
    elif distance_m <= 80:
        score += 0.5

    # heading contribution
    if heading_delta_deg <= 15:
        score += 2.0
    elif heading_delta_deg <= 30:
        score += 1.5
    elif heading_delta_deg <= 45:
        score += 1.0
    elif heading_delta_deg <= 60:
        score += 0.5

    # mild penalty for being far
    score -= min(distance_m / 200.0, 0.6)

    return round(score, 4)


def pass_geom_visibility(distance_m: float, heading_delta_deg: float, max_distance_m: float) -> bool:
    """Dynamic heading threshold + fixed max distance."""
    if distance_m > max_distance_m:
        return False
    return heading_delta_deg <= dynamic_heading_threshold_deg(distance_m)


def thin_sequence(images: List[Dict[str, Any]], min_move_m: float = 2.0, min_heading_change_deg: float = 5.0) -> List[Dict[str, Any]]:
    """Keep diverse frames; assumes images sorted by captured_at."""
    kept: List[Dict[str, Any]] = []
    last = None
    for img in images:
        if last is None:
            kept.append(img)
            last = img
            continue
        p = img.get("pose") or {}
        lp = last.get("pose") or {}
        d = haversine_m(p.get("lat"), p.get("lon"), lp.get("lat"), lp.get("lon"))
        dh = angle_diff_deg(p.get("heading", 0.0), lp.get("heading", 0.0))
        if d >= min_move_m or dh >= min_heading_change_deg:
            kept.append(img)
            last = img
    return kept


def spatial_join_one_coordinate(
    record: Dict[str, Any],
    *,
    max_distance_m: float = 50.0,
    top_k_candidates: int = 3,
    thin_per_facade: bool = True,
    max_keep_per_facade: int = 40,
) -> Dict[str, Any]:
    osm_raw = record.get("osm_raw") or record.get("osm")
    mapillary = record.get("mapillary", []) or []

    buildings = extract_osm_buildings(osm_raw) if osm_raw else []
    b_by_id = {b["osm_id"]: {**b, "assigned_images": []} for b in buildings}

    joined_images: List[Dict[str, Any]] = []

    for img in mapillary:
        pose = mapillary_pose(img)
        image_id = img.get("id")
        url = img.get("thumb_original_url")

        if pose is None or not buildings:
            joined_images.append({
                "image_id": image_id,
                "url": url,
                "pose": None if pose is None else {"lat": pose["lat"], "lon": pose["lon"], "heading": pose["heading"], "captured_at": pose["captured_at"]},
                "best_candidate": None,
                "candidates": [],
                "passes_geom_visibility": False,
                "reason": "no_pose_or_no_buildings",
            })
            continue

        cam_lat, cam_lon, heading = pose["lat"], pose["lon"], pose["heading"]
        captured_at = pose["captured_at"]

        # 1. NEW FILTER: DAYLIGHT CHECK
        if not is_approx_daylight(cam_lat, cam_lon, captured_at):
             joined_images.append({
                "image_id": image_id,
                "url": url,
                "pose": pose,
                "best_candidate": None,
                "candidates": [],
                "passes_geom_visibility": False,
                "reason": "darkness",
            })
             continue

        candidates = []
        for b in buildings:
            poly = b["footprint_latlon"]
            
            # 2. NEW FILTER: INSIDE CHECK
            # If camera is INSIDE the building polygon, skip it (likely GPS error or interior shot)
            if point_in_polygon(cam_lat, cam_lon, poly):
                continue

            # Edge-based metrics
            ei = nearest_edge_info(cam_lat, cam_lon, poly)
            if ei is None:
                continue

            # Bearing from camera to closest point on facade
            b_cam_to_cp = bearing_deg(cam_lat, cam_lon, ei.closest_point_lat, ei.closest_point_lon)
            heading_delta = angle_diff_deg(heading, b_cam_to_cp)
            
            # Updated score (no longer passes 'inside' var)
            sc = score_candidate(ei.distance_m, heading_delta)

            candidates.append({
                "osm_id": b["osm_id"],
                "distance_to_edge_m": round(ei.distance_m, 3),
                "closest_edge_index": ei.edge_index,
                "closest_point_latlon": [round(ei.closest_point_lat, 7), round(ei.closest_point_lon, 7)],
                "bearing_cam_to_closest_point_deg": round(b_cam_to_cp, 2),
                "heading_delta_deg": round(heading_delta, 2),
                "edge_bearing_deg": round(ei.edge_bearing_deg, 2),
                "facade_normal_deg": round(ei.facade_normal_deg, 2),
                "inside_polygon": False, # Always false now because we skip insides
                "visibility_score": round(sc, 3),
            })

        candidates.sort(key=lambda c: (-c["visibility_score"], c["distance_to_edge_m"]))
        candidates = candidates[:max(1, top_k_candidates)]

        best = candidates[0] if candidates else None
        if best is None:
            joined_images.append({
                "image_id": image_id,
                "url": url,
                "pose": {"lat": cam_lat, "lon": cam_lon, "heading": heading, "captured_at": pose["captured_at"]},
                "best_candidate": None,
                "candidates": [],
                "passes_geom_visibility": False,
                "reason": "no_candidates_or_inside_all",
            })
            continue

        passes = pass_geom_visibility(best["distance_to_edge_m"], best["heading_delta_deg"], max_distance_m=max_distance_m)

        join_info = {
            "image_id": image_id,
            "url": url,
            "pose": {"lat": cam_lat, "lon": cam_lon, "heading": heading, "captured_at": pose["captured_at"]},
            "best_candidate": best,          # includes osm_id + facade fields + score
            "candidates": candidates,        # top-K alternatives
            "assigned_building": best["osm_id"] if passes else None,
            "passes_geom_visibility": bool(passes),
            "reason": "geom_pass" if passes else "geom_reject",
        }

        if passes:
            b_by_id[best["osm_id"]]["assigned_images"].append(join_info)

        joined_images.append(join_info)

    # Optional: thin near-duplicates per building+facade
    if thin_per_facade:
        for bid, b in b_by_id.items():
            imgs = b["assigned_images"]
            # group by facade edge index
            by_edge: Dict[int, List[Dict[str, Any]]] = {}
            for it in imgs:
                edge = (it.get("best_candidate") or {}).get("closest_edge_index")
                if edge is None:
                    continue
                by_edge.setdefault(int(edge), []).append(it)

            new_imgs = []
            for edge, group in by_edge.items():
                # sort by captured_at (fallback: keep original order)
                group.sort(key=lambda x: (x.get("pose", {}).get("captured_at") or 0))
                thinned = thin_sequence(group, min_move_m=2.0, min_heading_change_deg=5.0)
                # cap per facade by score
                thinned.sort(key=lambda x: -((x.get("best_candidate") or {}).get("visibility_score") or 0))
                thinned = thinned[:max_keep_per_facade]
                new_imgs.extend(thinned)

            # Replace assigned_images with thinned set
            b["assigned_images"] = new_imgs

    # Add per-building summaries (handoff-friendly)
    for bid, b in b_by_id.items():
        imgs = b["assigned_images"]
        b["summary"] = {
            "num_images_after_filters": len(imgs),
            "num_images_before_thinning": None,  # could be added if you want to preserve it
            "images_by_facade_edge_index": {},
            "best_images": [],
        }

        counts: Dict[int, int] = {}
        for it in imgs:
            edge = (it.get("best_candidate") or {}).get("closest_edge_index")
            if edge is None:
                continue
            counts[int(edge)] = counts.get(int(edge), 0) + 1
        b["summary"]["images_by_facade_edge_index"] = counts

        # top 10 best
        best_sorted = sorted(imgs, key=lambda x: -((x.get("best_candidate") or {}).get("visibility_score") or 0))
        b["summary"]["best_images"] = [
            {
                "image_id": x.get("image_id"),
                "url": x.get("url"),
                "score": (x.get("best_candidate") or {}).get("visibility_score"),
                "edge": (x.get("best_candidate") or {}).get("closest_edge_index"),
                "distance_m": (x.get("best_candidate") or {}).get("distance_to_edge_m"),
                "heading_delta_deg": (x.get("best_candidate") or {}).get("heading_delta_deg"),
            }
            for x in best_sorted[:10]
        ]

    record_out = dict(record)
    record_out["spatial_join_params"] = {
        "max_distance_m": max_distance_m,
        "top_k_candidates": top_k_candidates,
        "dynamic_heading_threshold": "15m->60deg, 30m->45deg, 50m->30deg, else 25deg",
        "thinning": {
            "enabled": thin_per_facade,
            "min_move_m": 2.0,
            "min_heading_change_deg": 5.0,
            "max_keep_per_facade": max_keep_per_facade,
        },
        "filters": {
             "remove_inside_building": True,
             "remove_dark_photos": True
        },
        "policy": [
            "compute top-K building candidates per image (distance+heading+inside)",
            "assign only if passes geom visibility gate",
            "group by building+facade edge; optional sequence thinning",
        ],
    }
    record_out["buildings_joined"] = list(b_by_id.values())
    record_out["mapillary_joined"] = joined_images
    return record_out


def main():
    inp = Path("per_coordinate_osm_mapillary.json")
    out = Path("per_coordinate_spatial_join.json")

    print(">>> reading", inp.resolve())
    data = json.loads(inp.read_text())

    print(">>> records:", len(data))
    joined = [spatial_join_one_coordinate(r) for r in data]

    out.write_text(json.dumps(joined, indent=2))
    print(">>> wrote", out.resolve())


if __name__ == "__main__":
    main()