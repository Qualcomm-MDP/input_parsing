import os
import json
import time
import requests
from dotenv import load_dotenv

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_URL = "https://graph.mapillary.com/images"


def bbox_from_center(lat, lon, buffer_deg):
    return (lat - buffer_deg, lon - buffer_deg, lat + buffer_deg, lon + buffer_deg)


def overpass_query(bbox):
    s, w, n, e = bbox
    return f"""
[out:json][timeout:25];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out body;
>;
out skel qt;
"""


def fetch_osm(bbox):
    r = requests.post(OVERPASS_URL, data={"data": overpass_query(bbox)}, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_mapillary(bbox, token, limit=200):
    s, w, n, e = bbox
    bbox_str = f"{w},{s},{e},{n}"
    fields = "id,thumb_original_url,computed_geometry,computed_compass_angle,camera_parameters,captured_at"

    params = {
        "bbox": bbox_str,
        "fields": fields,
        "access_token": token,
        "limit": limit,
    }

    r = requests.get(MAPILLARY_URL, params=params, timeout=30)
    if r.status_code != 200:
        return {"error": r.text}

    return r.json().get("data", [])


def main():
    load_dotenv()
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing")

    coords = [
        (42.2912, -83.7175),
        # add more coordinates here
    ]

    buffer_deg = 0.001
    results = []

    for lat, lon in coords:
        bbox = bbox_from_center(lat, lon, buffer_deg)

        record = {
            "coordinate": {"lat": lat, "lon": lon},
            "bbox_south_west_north_east": bbox,
            "osm_raw": fetch_osm(bbox),
            "mapillary": fetch_mapillary(bbox, token),
        }

        results.append(record)
        time.sleep(0.25)

    with open("per_coordinate_osm_mapillary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved per_coordinate_osm_mapillary.json")


if __name__ == "__main__":
    main()
