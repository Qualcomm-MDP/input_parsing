print(">>> LOADED:", __file__)
import json
import math
from pathlib import Path


# -----------------------------
# Geometry helpers
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def point_in_polygon(lat, lon, polygon_latlon):
    """
    Ray casting. polygon_latlon = [(lat, lon), ...]
    """
    x, y = lon, lat
    inside = False
    n = len(polygon_latlon)
    for i in range(n):
        y1, x1 = polygon_latlon[i]
        y2, x2 = polygon_latlon[(i + 1) % n]
        # Edge crosses horizontal ray?
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def point_to_segment_distance_m(lat, lon, a_lat, a_lon, b_lat, b_lon):
    """
    Approximate distance point->segment in meters using a local projection:
    - Convert lat/lon to meters in a small neighborhood (good for city scale).
    """
    # Local meters-per-degree
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * math.cos(lat_rad)

    def to_xy(phi, lam):
        return (lam * m_per_deg_lon, phi * m_per_deg_lat)

    px, py = to_xy(lat, lon)
    ax, ay = to_xy(a_lat, a_lon)
    bx, by = to_xy(b_lat, b_lon)

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        # a==b
        dx, dy = px - ax, py - ay
        return math.hypot(dx, dy)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def point_to_polygon_distance_m(lat, lon, polygon_latlon):
    """
    Min distance to polygon edges (meters).
    polygon_latlon = [(lat, lon), ...]
    """
    best = float("inf")
    n = len(polygon_latlon)
    for i in range(n):
        a_lat, a_lon = polygon_latlon[i]
        b_lat, b_lon = polygon_latlon[(i + 1) % n]
        d = point_to_segment_distance_m(lat, lon, a_lat, a_lon, b_lat, b_lon)
        if d < best:
            best = d
    return best


# -----------------------------
# OSM extraction
# -----------------------------

def extract_osm_buildings(overpass_json):
    """
    Returns list of buildings with:
      - osm_id
      - tags
      - footprint_latlon = [(lat, lon), ...]
    Only supports building 'ways' (your data is mostly ways).
    """
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

        buildings.append({
            "osm_id": e.get("id"),
            "tags": tags,
            "footprint_latlon": footprint,
        })

    return buildings


# -----------------------------
# Spatial Join
# -----------------------------

def spatial_join_one_coordinate(record, max_assign_distance_m=60.0):
    """
    record contains:
      - osm_raw
      - mapillary (either list of formatted entries OR raw Mapillary objects)

    Output adds:
      - buildings_joined: buildings with assigned_images
      - mapillary_joined: images with assigned_building (or None)
    """
    osm_raw = record.get("osm_raw") or record.get("osm")  # support either key
    mapillary = record.get("mapillary", [])

    buildings = extract_osm_buildings(osm_raw) if osm_raw else []
    # index by id for assignments
    b_by_id = {b["osm_id"]: {**b, "assigned_images": []} for b in buildings}

    joined_images = []
    for img in mapillary:
        # Support your formatted mapillary entries OR raw Mapillary response objects
        if "pose" in img:
            lat = img["pose"].get("lat")
            lon = img["pose"].get("lon")
            heading = img["pose"].get("heading")
            image_id = img.get("image_id")
            url = img.get("url")
        else:
            # raw Mapillary "data" item
            coords = img.get("computed_geometry", {}).get("coordinates", [None, None])
            lon, lat = (coords[0], coords[1]) if len(coords) >= 2 else (None, None)
            heading = img.get("computed_compass_angle")
            image_id = img.get("id")
            url = img.get("thumb_original_url")

        if lat is None or lon is None or not buildings:
            joined_images.append({
                "image_id": image_id,
                "url": url,
                "pose": {"lat": lat, "lon": lon, "heading": heading},
                "assigned_building": None,
                "reason": "no_pose_or_no_buildings",
            })
            continue

        # 1) Prefer building that CONTAINS the point
        containing = None
        for b in buildings:
            if point_in_polygon(lat, lon, b["footprint_latlon"]):
                containing = b
                break

        if containing:
            bid = containing["osm_id"]
            join_info = {
                "image_id": image_id,
                "url": url,
                "pose": {"lat": lat, "lon": lon, "heading": heading},
                "assigned_building": bid,
                "distance_to_building_m": 0.0,
                "reason": "inside_polygon",
            }
            b_by_id[bid]["assigned_images"].append(join_info)
            joined_images.append(join_info)
            continue

        # 2) Else choose nearest building by distance to polygon edges
        best_bid = None
        best_d = float("inf")
        for b in buildings:
            d = point_to_polygon_distance_m(lat, lon, b["footprint_latlon"])
            if d < best_d:
                best_d = d
                best_bid = b["osm_id"]

        if best_bid is not None and best_d <= max_assign_distance_m:
            join_info = {
                "image_id": image_id,
                "url": url,
                "pose": {"lat": lat, "lon": lon, "heading": heading},
                "assigned_building": best_bid,
                "distance_to_building_m": round(best_d, 3),
                "reason": "nearest_edge",
            }
            b_by_id[best_bid]["assigned_images"].append(join_info)
            joined_images.append(join_info)
        else:
            joined_images.append({
                "image_id": image_id,
                "url": url,
                "pose": {"lat": lat, "lon": lon, "heading": heading},
                "assigned_building": None,
                "distance_to_building_m": None,
                "reason": "too_far_or_no_buildings",
            })

    record_out = dict(record)
    record_out["spatial_join_params"] = {
        "max_assign_distance_m": max_assign_distance_m,
        "policy": ["inside_polygon_first", "else_nearest_polygon_edge"],
    }
    record_out["buildings_joined"] = list(b_by_id.values())
    record_out["mapillary_joined"] = joined_images
    return record_out


def main():
    try:
        print(">>> spatial join script started")

        inp = Path("per_coordinate_osm_mapillary.json")
        out = Path("per_coordinate_spatial_join.json")

        print(">>> reading", inp.resolve())
        data = json.loads(inp.read_text())

        print(">>> records:", len(data))
        joined = [
            spatial_join_one_coordinate(r, max_assign_distance_m=60.0)
            for r in data
        ]  # IMPORTANT: NO trailing ()

        out.write_text(json.dumps(joined, indent=2))
        print(">>> wrote", out.resolve())

    except Exception as e:
        print(">>> ERROR:", repr(e))
        raise

if __name__ == "__main__":
    main()
