#!/usr/bin/env python3
print(">>> LOADED:", __file__)

import json
import math
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# -----------------------------
# CONFIG
# -----------------------------
THUMB_DIR = Path("review_thumbnails")
THUMB_DIR.mkdir(parents=True, exist_ok=True)

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


def polygon_area_m2(polygon_latlon: List[Tuple[float, float]]) -> float:
    """Approx polygon area (m^2) in local coords around polygon centroid."""
    if len(polygon_latlon) < 3:
        return 0.0

    # If closed, drop last
    poly = polygon_latlon
    if len(poly) >= 4 and poly[0] == poly[-1]:
        poly = poly[:-1]

    if len(poly) < 3:
        return 0.0

    lat0 = sum(p[0] for p in poly) / len(poly)
    lon0 = sum(p[1] for p in poly) / len(poly)
    pts = [latlon_to_local_xy(lat, lon, lat0, lon0) for lat, lon in poly]
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
    # Remove closure if present so we can re-add properly
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    if len(cleaned) >= 3:
        if cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0])
    return cleaned


def point_in_polygon(lat: float, lon: float, polygon_latlon: List[Tuple[float, float]]) -> bool:
    """Ray casting. polygon_latlon should be closed."""
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


def point_in_polygon_with_holes(
    lat: float, lon: float,
    outer_ring_closed: List[Tuple[float, float]],
    inner_rings_closed: Optional[List[List[Tuple[float, float]]]] = None
) -> bool:
    """Inside outer AND not inside any inner ring."""
    if not outer_ring_closed or len(outer_ring_closed) < 4:
        return False
    if not point_in_polygon(lat, lon, outer_ring_closed):
        return False
    if inner_rings_closed:
        for hole in inner_rings_closed:
            if hole and len(hole) >= 4 and point_in_polygon(lat, lon, hole):
                return False
    return True


def is_approx_daylight(lat: float, lon: float, captured_at_ms: Optional[int]) -> bool:
    """
    Checks if photo was likely taken during the day.
    Uses approx local solar time (UTC + lon/15).
    Rejects if before 7am or after 7pm local solar time.
    """
    if captured_at_ms is None:
        return True  # Default to keep if no time data

    try:
        dt_utc = datetime.fromtimestamp(captured_at_ms / 1000.0, tz=timezone.utc)
        hour_offset = lon / 15.0
        local_hour = (dt_utc.hour + (dt_utc.minute / 60.0) + hour_offset) % 24
        return 7.0 <= local_hour <= 19.0
    except Exception:
        return True


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

    e_bearing = bearing_deg(a_lat, a_lon, b_lat, b_lon)

    n1 = wrap_angle_deg(e_bearing + 90)
    n2 = wrap_angle_deg(e_bearing - 90)
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
    h = tags.get("height")
    if isinstance(h, str):
        s = h.strip().lower().replace("meters", "m").replace(" ", "")
        try:
            if s.endswith("m"):
                s = s[:-1]
            return float(s)
        except Exception:
            pass

    lv = tags.get("building:levels")
    if isinstance(lv, str):
        try:
            levels = float(lv.strip())
            return max(1.0, levels) * 3.0
        except Exception:
            pass

    return None

# -----------------------------
# Download Helper
# -----------------------------
def download_thumbnail(url: str, image_id: str) -> Optional[str]:
    """
    Downloads the thumbnail to THUMB_DIR and returns the local path string.
    Returns None if download failed.
    """
    if not url: 
        return None
    
    filename = THUMB_DIR / f"{image_id}.jpg"
    
    # Don't re-download if we already have it
    if filename.exists():
        return str(filename)
        
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            filename.write_bytes(r.content)
            time.sleep(0.1)
            return str(filename)
        else:
            print(f"xx Error {r.status_code} downloading {image_id}")
    except Exception as e:
        print(f"xx Failed to download {image_id}: {e}")
    return None

# -----------------------------
# OSM extraction (ways + relations)
# -----------------------------

def _way_nodes_latlon(way: Dict[str, Any], nodes_by_id: Dict[int, Tuple[float, float]]) -> List[Tuple[float, float]]:
    footprint = [nodes_by_id.get(nid) for nid in (way.get("nodes") or [])]
    footprint = [p for p in footprint if p and p[0] is not None and p[1] is not None]
    return footprint


def _stitch_rings_from_way_node_lists(
    way_node_lists: List[List[Tuple[float, float]]]
) -> List[List[Tuple[float, float]]]:
    segments = [seg for seg in way_node_lists if seg and len(seg) >= 2]
    rings: List[List[Tuple[float, float]]] = []

    while segments:
        ring = segments.pop(0)[:]

        changed = True
        while changed and segments:
            changed = False
            end = ring[-1]
            start = ring[0]

            if len(ring) >= 3 and ring[0] == ring[-1]:
                break

            for i, seg in enumerate(segments):
                s0, s1 = seg[0], seg[-1]

                if end == s0:
                    ring.extend(seg[1:])
                    segments.pop(i)
                    changed = True
                    break
                if end == s1:
                    ring.extend(list(reversed(seg[:-1])))
                    segments.pop(i)
                    changed = True
                    break
                if start == s1:
                    ring = seg[:-1] + ring
                    segments.pop(i)
                    changed = True
                    break
                if start == s0:
                    ring = list(reversed(seg[1:])) + ring
                    segments.pop(i)
                    changed = True
                    break

        ring = clean_polygon_latlon(ring)
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)

    return rings


def _extract_relation_multipolygon_rings(
    rel: Dict[str, Any],
    ways_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Tuple[float, float]]
) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    outers: List[List[Tuple[float, float]]] = []
    inners: List[List[Tuple[float, float]]] = []

    members = rel.get("members") or []
    outer_way_lists: List[List[Tuple[float, float]]] = []
    inner_way_lists: List[List[Tuple[float, float]]] = []

    for m in members:
        if m.get("type") != "way":
            continue
        wid = m.get("ref")
        role = (m.get("role") or "").strip().lower()
        way = ways_by_id.get(wid)
        if not way:
            continue

        pts = _way_nodes_latlon(way, nodes_by_id)
        if len(pts) < 2:
            continue

        if role == "inner":
            inner_way_lists.append(pts)
        else:
            outer_way_lists.append(pts)

    if outer_way_lists:
        outers = _stitch_rings_from_way_node_lists(outer_way_lists)
    if inner_way_lists:
        inners = _stitch_rings_from_way_node_lists(inner_way_lists)

    return outers, inners


def extract_osm_buildings(overpass_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    elements = overpass_json.get("elements", []) if isinstance(overpass_json, dict) else []

    nodes_by_id: Dict[int, Tuple[float, float]] = {
        e["id"]: (e["lat"], e["lon"])
        for e in elements
        if e.get("type") == "node" and e.get("id") is not None
    }

    ways_by_id: Dict[int, Dict[str, Any]] = {
        e["id"]: e
        for e in elements
        if e.get("type") == "way" and e.get("id") is not None
    }

    buildings: List[Dict[str, Any]] = []

    for e in elements:
        if e.get("type") != "way":
            continue
        tags = e.get("tags", {}) or {}
        if "building" not in tags:
            continue

        footprint = _way_nodes_latlon(e, nodes_by_id)
        if len(footprint) < 3:
            continue
        footprint = clean_polygon_latlon(footprint)
        if len(footprint) < 4:
            continue
        if polygon_area_m2(footprint) < 5.0:
            continue

        verts = footprint[:-1]
        c_lat = sum(p[0] for p in verts) / len(verts)
        c_lon = sum(p[1] for p in verts) / len(verts)

        buildings.append({
            "osm_id": f"way/{e.get('id')}",
            "osm_type": "way",
            "tags": tags,
            "height_m": parse_height_m(tags),
            "footprint_latlon": footprint,
            "holes_latlon": [],
            "centroid_latlon": (c_lat, c_lon),
        })

    for e in elements:
        if e.get("type") != "relation":
            continue
        tags = e.get("tags", {}) or {}
        if "building" not in tags:
            continue

        outer_rings, inner_rings = _extract_relation_multipolygon_rings(e, ways_by_id, nodes_by_id)
        if not outer_rings:
            continue

        outer_rings_sorted = sorted(outer_rings, key=lambda r: polygon_area_m2(r), reverse=True)
        primary_outer = outer_rings_sorted[0]

        if len(primary_outer) < 4 or polygon_area_m2(primary_outer) < 5.0:
            continue

        verts = primary_outer[:-1]
        c_lat = sum(p[0] for p in verts) / len(verts)
        c_lon = sum(p[1] for p in verts) / len(verts)

        valid_holes: List[List[Tuple[float, float]]] = []
        for hole in inner_rings:
            if len(hole) < 4:
                continue
            hv = hole[:-1]
            hlat = sum(p[0] for p in hv) / len(hv)
            hlon = sum(p[1] for p in hv) / len(hv)
            if point_in_polygon(hlat, hlon, primary_outer):
                valid_holes.append(hole)

        buildings.append({
            "osm_id": f"relation/{e.get('id')}",
            "osm_type": "relation",
            "tags": tags,
            "height_m": parse_height_m(tags),
            "footprint_latlon": primary_outer,
            "holes_latlon": valid_holes,
            "centroid_latlon": (c_lat, c_lon),
            "outer_rings_latlon": outer_rings_sorted,
        })

    return buildings


# -----------------------------
# Join logic
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
    score = 0.0
    if distance_m <= 15: score += 2.0
    elif distance_m <= 30: score += 1.5
    elif distance_m <= 50: score += 1.0
    elif distance_m <= 80: score += 0.5

    if heading_delta_deg <= 15: score += 2.0
    elif heading_delta_deg <= 30: score += 1.5
    elif heading_delta_deg <= 45: score += 1.0
    elif heading_delta_deg <= 60: score += 0.5

    score -= min(distance_m / 200.0, 0.6)
    return round(score, 4)


def pass_geom_visibility(distance_m: float, heading_delta_deg: float, max_distance_m: float) -> bool:
    if distance_m > max_distance_m:
        return False
    return heading_delta_deg <= dynamic_heading_threshold_deg(distance_m)


def thin_sequence(images: List[Dict[str, Any]], min_move_m: float = 2.0, min_heading_change_deg: float = 5.0) -> List[Dict[str, Any]]:
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

    kept_images: List[Dict[str, Any]] = []
    discarded_urls: List[str] = []

    print(f">> Processing batch with {len(mapillary)} images...")

    for img in mapillary:
        pose = mapillary_pose(img)
        image_id = img.get("id")
        url = img.get("thumb_original_url")
        if not url:
            continue

        # Check 1: Valid Pose & Buildings
        if pose is None or not buildings:
            discarded_urls.append(url)
            continue

        cam_lat, cam_lon, heading = pose["lat"], pose["lon"], pose["heading"]
        captured_at = pose["captured_at"]

        # Check 2: Daylight
        if not is_approx_daylight(cam_lat, cam_lon, captured_at):
            discarded_urls.append(url)
            continue

        candidates = []
        for b in buildings:
            outer = b["footprint_latlon"]
            holes = b.get("holes_latlon") or []

            # Check 3: Inside Building
            if point_in_polygon_with_holes(cam_lat, cam_lon, outer, holes):
                continue

            ei = nearest_edge_info(cam_lat, cam_lon, outer)
            if ei is None:
                continue

            b_cam_to_cp = bearing_deg(cam_lat, cam_lon, ei.closest_point_lat, ei.closest_point_lon)
            heading_delta = angle_diff_deg(heading, b_cam_to_cp)
            sc = score_candidate(ei.distance_m, heading_delta)

            candidates.append({
                "osm_id": b["osm_id"],
                "osm_type": b.get("osm_type"),
                "distance_to_edge_m": round(ei.distance_m, 3),
                "closest_edge_index": ei.edge_index,
                "closest_point_latlon": [round(ei.closest_point_lat, 7), round(ei.closest_point_lon, 7)],
                "bearing_cam_to_closest_point_deg": round(b_cam_to_cp, 2),
                "heading_delta_deg": round(heading_delta, 2),
                "edge_bearing_deg": round(ei.edge_bearing_deg, 2),
                "facade_normal_deg": round(ei.facade_normal_deg, 2),
                "inside_polygon": False,
                "visibility_score": round(sc, 3),
            })

        candidates.sort(key=lambda c: (-c["visibility_score"], c["distance_to_edge_m"]))
        candidates = candidates[:max(1, top_k_candidates)]

        best = candidates[0] if candidates else None
        
        # Check 4: No Candidates (or all were inside)
        if best is None:
            discarded_urls.append(url)
            continue

        passes = pass_geom_visibility(best["distance_to_edge_m"], best["heading_delta_deg"], max_distance_m=max_distance_m)

        # Check 5: Geometry Pass
        if not passes:
            discarded_urls.append(url)
            continue

        # --- PASSED ALL CHECKS ---
        print(f"   [+] Downloading {image_id}...")
        local_thumb_path = download_thumbnail(url, image_id)

        join_info = {
            "image_id": image_id,
            "url": url,
            "pose": {"lat": cam_lat, "lon": cam_lon, "heading": heading, "captured_at": pose["captured_at"]},
            "best_candidate": best,
            "candidates": candidates,
            "assigned_building": best["osm_id"],
            "passes_geom_visibility": True,
            "local_thumb_path": local_thumb_path,
            "reason": "geom_pass",
        }

        b_by_id[best["osm_id"]]["assigned_images"].append(join_info)
        kept_images.append(join_info)

    # Thinning logic (unchanged)
    if thin_per_facade:
        for bid, b in b_by_id.items():
            imgs = b["assigned_images"]
            by_edge: Dict[int, List[Dict[str, Any]]] = {}
            for it in imgs:
                edge = (it.get("best_candidate") or {}).get("closest_edge_index")
                if edge is None:
                    continue
                by_edge.setdefault(int(edge), []).append(it)

            new_imgs = []
            for edge, group in by_edge.items():
                group.sort(key=lambda x: (x.get("pose", {}).get("captured_at") or 0))
                thinned = thin_sequence(group, min_move_m=2.0, min_heading_change_deg=5.0)
                thinned.sort(key=lambda x: -((x.get("best_candidate") or {}).get("visibility_score") or 0))
                thinned = thinned[:max_keep_per_facade]
                new_imgs.extend(thinned)

            b["assigned_images"] = new_imgs

    # Add summaries
    for bid, b in b_by_id.items():
        imgs = b["assigned_images"]
        b["summary"] = {
            "num_images_after_filters": len(imgs),
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

        best_sorted = sorted(imgs, key=lambda x: -((x.get("best_candidate") or {}).get("visibility_score") or 0))
        b["summary"]["best_images"] = [
            {
                "image_id": x.get("image_id"),
                "url": x.get("url"),
                "local_thumb_path": x.get("local_thumb_path"),
                "score": (x.get("best_candidate") or {}).get("visibility_score"),
                "edge": (x.get("best_candidate") or {}).get("closest_edge_index"),
                "distance_m": (x.get("best_candidate") or {}).get("distance_to_edge_m"),
                "heading_delta_deg": (x.get("best_candidate") or {}).get("heading_delta_deg"),
                "osm_id": (x.get("best_candidate") or {}).get("osm_id"),
            }
            for x in best_sorted[:10]
        ]

    record_out = dict(record)
    record_out["spatial_join_params"] = {
        "max_distance_m": max_distance_m,
        "filters": {"remove_inside_building": True, "remove_dark_photos": True}
    }
    record_out["buildings_joined"] = list(b_by_id.values())
    record_out["mapillary_kept"] = kept_images
    
    # Remove the old raw mapillary list if desired, or keep it. 
    # For cleanliness, let's remove the raw list from the output so it's not huge
    if "mapillary" in record_out:
        del record_out["mapillary"]

    # Place discarded URLs at the very end
    record_out["discarded_mapillary_urls"] = discarded_urls
    
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
    print(f">>> Images saved to: {THUMB_DIR.resolve()}")


if __name__ == "__main__":
    main()