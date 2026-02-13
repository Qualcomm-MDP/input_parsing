#!/usr/bin/env python3
import os
import json
import time
import argparse
import requests
from dotenv import load_dotenv

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_URL = "https://graph.mapillary.com/images"


def parse_coord(s):
    lat_s, lon_s = s.split(",")
    return float(lat_s), float(lon_s)


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

    params = {
        "bbox": bbox_str,
        "fields": "id,thumb_original_url,computed_geometry,computed_compass_angle,captured_at",
        "access_token": token,
        "limit": limit,
    }

    r = requests.get(MAPILLARY_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--coord", action="append", required=True,
                        help="Coordinate as lat,lon (can repeat)")
    parser.add_argument("--buffer", type=float, default=0.001,
                        help="BBox buffer in degrees (default 0.001)")
    parser.add_argument("-o", "--out", type=str,
                        help="Output file (optional)")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing")

    results = []

    for coord_str in args.coord:
        lat, lon = parse_coord(coord_str)
        bbox = bbox_from_center(lat, lon, args.buffer)

        record = {
            "coordinate": {"lat": lat, "lon": lon},
            "bbox_south_west_north_east": bbox,
            "osm_raw": fetch_osm(bbox),
            "mapillary": fetch_mapillary(bbox, token),
        }

        results.append(record)
        time.sleep(0.25)

    output_json = json.dumps(results, indent=2)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output_json)
        print(f"Saved {args.out}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()