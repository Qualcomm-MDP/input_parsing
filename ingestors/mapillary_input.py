import os
import json
import time
import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon
from collections import defaultdict
from dotenv import load_dotenv

# API Endpoints
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_URL = "https://graph.mapillary.com/images"

class MapillaryIngestor:
    def __init__(self, token=None, base_dir="data"):
        load_dotenv()
        self.token = token or os.getenv("MAPILLARY_ACCESS_TOKEN")
        self.headers = {"Authorization": f"OAuth {self.token}"}
        self.manifest = []
        self.osm_buildings = []
        
        self.base_dir = base_dir
        self.assets_dir = os.path.join(self.base_dir, "mapillary")
        self.manifest_path = os.path.join(self.base_dir, "manifest.json")
        self.osm_path = os.path.join(self.base_dir, "osm_buildings.json")
        self._setup_directories()

    def _setup_directories(self):
        """Ensures the data folder structure exists."""
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
            print(f"Created directory: {self.assets_dir}")

    def run_pipeline(self, lat, lon, buffer=0.001):
        """Orchestrates fetch and download for Mapillary and OSM."""
        print(f"--- Starting Input Pipeline: ({lat}, {lon}) ---")
        
        self._fetch_metadata(lat, lon, buffer)
        self._fetch_osm_buildings(lat, lon, buffer)
        
        if not self.manifest and not self.osm_buildings:
            print("[ERROR] No data found for this area.")
            return

        self._save_manifest()
        self._save_osm_metadata()
        self._download_images()
        
        print(f"--- Pipeline Complete ---")

    def _fetch_metadata(self, lat, lon, buffer):
        """Fetches Mapillary metadata including sequence IDs."""
        bbox = f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer}"
        fields = "id,thumb_original_url,computed_geometry,computed_compass_angle,camera_parameters,captured_at,sequence"
        url = f"{MAPILLARY_URL}?bbox={bbox}&fields={fields}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                raw_data = response.json().get("data", [])
                self.manifest = [self._format_entry(img) for img in raw_data]
            else:
                print(f"[ERROR] Mapillary API {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[CRITICAL] Mapillary fetch failed: {e}")

    def _format_entry(self, img):
        """Packs metadata into a structured format."""
        return {
            "image_id": img['id'],
            "sequence_id": img.get('sequence'),
            "url": img.get('thumb_original_url'),
            "captured_at": img.get('captured_at'),
            "pose": {
                "lat": img['computed_geometry']['coordinates'][1],
                "lon": img['computed_geometry']['coordinates'][0],
                "heading": img.get('computed_compass_angle')
            }
        }

    def _fetch_osm_buildings(self, lat, lon, buffer, retries=3):
        """Queries Overpass API for buildings with retry logic."""
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

    def _download_images(self):
        """Downloads images into sequence-specific subdirectories."""
        sequences = defaultdict(list)
        for img in self.manifest:
            sid = img.get('sequence_id', 'unknown')
            sequences[sid].append(img)

        for sid, images in sequences.items():
            seq_dir = os.path.join(self.assets_dir, str(sid))
            os.makedirs(seq_dir, exist_ok=True)
            for img in images:
                file_path = os.path.join(seq_dir, f"{img['image_id']}.jpg")
                if not os.path.exists(file_path):
                    try:
                        r = requests.get(img['url'], stream=True)
                        if r.status_code == 200:
                            with open(file_path, "wb") as f:
                                for chunk in r.iter_content(1024): f.write(chunk)
                    except Exception as e: print(f"Error downloading {img['image_id']}: {e}")

def export_spatial_manifest(base_dir="data"):
    """Performs spatial join and saves formatted GeoJSON."""
    manifest_path = os.path.join(base_dir, "manifest.json")
    osm_path = os.path.join(base_dir, "osm_buildings.json")

    with open(manifest_path, "r") as f: manifest = json.load(f)
    with open(osm_path, "r") as f: osm_elements = json.load(f)

    # Process Points
    img_list = [{
        "image_id": i['image_id'], "sequence_id": i['sequence_id'],
        "lat": i['pose']['lat'], "lon": i['pose']['lon'], "heading": i['pose']['heading'],
        "geometry": Point(i['pose']['lon'], i['pose']['lat'])
    } for i in manifest]
    images_gdf = gpd.GeoDataFrame(img_list, crs="EPSG:4326")

    # Process Polygons
    nodes = {el['id']: (el['lon'], el['lat']) for el in osm_elements if el['type'] == 'node'}
    building_list = []
    for el in osm_elements:
        if el['type'] == 'way' and 'nodes' in el:
            coords = [nodes[nid] for nid in el['nodes'] if nid in nodes]
            if len(coords) >= 3:
                building_list.append({
                    "osm_id": el['id'], "building_name": el.get('tags', {}).get('name', 'Unknown'),
                    "geometry": Polygon(coords)
                })
    buildings_gdf = gpd.GeoDataFrame(building_list, crs="EPSG:4326")

    # Spatial Join
    res = gpd.sjoin_nearest(images_gdf.to_crs(3857), buildings_gdf.to_crs(3857), distance_col="distance_m")
    
    # Reorder for readability
    cols = ['building_name', 'distance_m', 'image_id', 'sequence_id', 'heading', 'lat', 'lon', 'osm_id', 'geometry']
    final_gdf = res[cols].to_crs(4326)

    # Indented Save
    output_path = os.path.join(base_dir, "spatial_manifest.json")
    with open(output_path, "w") as f:
        json.dump(json.loads(final_gdf.to_json()), f, indent=4)
    print(f"Saved manifest to {output_path}")

if __name__ == "__main__":
    ingestor = MapillaryIngestor()
    ingestor.run_pipeline(42.2912, -83.7175)
    export_spatial_manifest()