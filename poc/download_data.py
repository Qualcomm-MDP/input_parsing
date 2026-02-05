import os
import json
import time
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from collections import defaultdict
from dotenv import load_dotenv

# API Endpoints
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_URL = "https://graph.mapillary.com/images"

class Ingestor:
    def __init__(self, token=None, base_dir="data"):
        load_dotenv()
        self.token = token or os.getenv("MAPILLARY_ACCESS_TOKEN")
        self.headers = {"Authorization": f"OAuth {self.token}"}
        self.manifest = []
        self.osm_buildings = []
        
        self.base_dir = base_dir
        # [FIX] Consistent Assets Directory
        self.assets_dir = os.path.join(self.base_dir, "assets")
        self.manifest_path = os.path.join(self.base_dir, "mapillary_images.json")
        self.osm_path = os.path.join(self.base_dir, "osm_buildings.json")
        self._setup_directories()

    def _setup_directories(self):
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)

    def run_pipeline(self, lat, lon, buffer=0.001):
            print(f"--- Starting Input Pipeline: ({lat}, {lon}) ---")
            
            self._fetch_metadata(lat, lon, buffer)
            self._fetch_osm_buildings(lat, lon, buffer)
            
            if not self.manifest and not self.osm_buildings:
                print("[ERROR] No data found.")
                return

            self._save_manifest()
            self._save_osm_metadata()
            
            # [CHANGE 1] REMOVED self._download_images() 
            # We now only save metadata. Images come later.
            
            print(f"--- Metadata Complete (Images deferred) ---")

    # [CHANGE 2] NEW METHOD: The Lazy Downloader
    def download_matches(self, matches_parquet_path):
        """Downloads only the images that were successfully matched."""
        if not os.path.exists(matches_parquet_path):
            print("No matches file found.")
            return

        try:
            # Requires pandas to read the parquet matches
            import pandas as pd
            df = pd.read_parquet(matches_parquet_path)
        except Exception as e:
            print(f"Error reading matches: {e}")
            return

        print(f"Hydrating {len(df)} matched assets...")
        
        count = 0
        for _, row in df.iterrows():
            # Get the destination path
            rel_path = row.get('image_path')
            url = row.get('url')
            
            if not rel_path or not url: continue

            # Construct absolute path
            full_path = os.path.abspath(rel_path)
            
            # check if exists
            if os.path.exists(full_path):
                count += 1
                continue
                
            # create dir
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # download
            try:
                r = requests.get(url, stream=True, timeout=10)
                if r.status_code == 200:
                    with open(full_path, "wb") as f:
                        for chunk in r.iter_content(1024): f.write(chunk)
                    count += 1
            except Exception as e:
                print(f"Failed to download {rel_path}: {e}")
        
        print(f"Hydration complete: {count}/{len(df)} images ready.")

    def _fetch_metadata(self, lat, lon, buffer):
        bbox = f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer}"
        fields = "id,thumb_original_url,computed_geometry,computed_compass_angle,camera_parameters,camera_type,captured_at,sequence"
        url = f"{MAPILLARY_URL}?bbox={bbox}&fields={fields}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                raw_data = response.json().get("data", [])
                self.manifest = [self._format_entry(img) for img in raw_data]
            else:
                print(f"Mapillary Error: {response.text}")
        except Exception as e:
            print(f"Fetch failed: {e}")

    def _format_entry(self, img):
            """Packs metadata into a structured format."""
            return {
                "image_id": img['id'],
                "sequence_id": img.get('sequence'),
                "url": img.get('thumb_original_url'),
                "captured_at": img.get('captured_at'),
                # [NEW] Save Camera Type
                "camera_type": img.get('camera_type', 'perspective'), 
                "pose": {
                    "lat": img['computed_geometry']['coordinates'][1],
                    "lon": img['computed_geometry']['coordinates'][0],
                    "heading": img.get('computed_compass_angle')
                }
            }

    def _fetch_osm_buildings(self, lat, lon, buffer, retries=3):
        s, w, n, e = (lat - buffer, lon - buffer, lat + buffer, lon + buffer)
        query = f"""
        [out:json][timeout:25];
        (
          way["building"]({s},{w},{n},{e});
          relation["building"]({s},{w},{n},{e});
        );
        out body;
        >;
        out skel qt;
        """
        for attempt in range(retries):
            try:
                print(f"Fetching OSM buildings (Attempt {attempt + 1})...")
                response = requests.post(OVERPASS_URL, data={"data": query}, timeout=60)
                if response.status_code == 200:
                    self.osm_buildings = response.json().get("elements", [])
                    return
                time.sleep((attempt + 1) * 2)
            except Exception as e:
                print(f"OSM attempt failed: {e}")

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=4)

    def _save_osm_metadata(self):
        with open(self.osm_path, "w") as f:
            json.dump(self.osm_buildings, f, indent=4)