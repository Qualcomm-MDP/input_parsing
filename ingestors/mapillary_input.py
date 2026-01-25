import os
import json
import requests
from dotenv import load_dotenv

class MapillaryIngestor:
    def __init__(self, token=None, base_dir="data"):
        load_dotenv()
        self.token = token or os.getenv("MAPILLARY_ACCESS_TOKEN")
        self.headers = {"Authorization": f"OAuth {self.token}"}
        self.manifest = []
        
        self.base_dir = base_dir
        self.assets_dir = os.path.join(self.base_dir, "mapillary")
        self.manifest_path = os.path.join(self.base_dir, "manifest.json")
        self._setup_directories()

    def _setup_directories(self):
        """Ensures the data folder structure exists."""
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
            print(f"Created directory: {self.assets_dir}")

    def run_pipeline(self, lat, lon, buffer=0.001):
        """Orchestrates the full fetch-to-download sequence."""
        print(f"--- Starting Input Pipeline: ({lat}, {lon}) ---")
        
        self._fetch_metadata(lat, lon, buffer)
        
        if not self.manifest:
            print("[ERROR] No data found for this area.")
            return

        self._save_manifest()
        self._download_images()
        
        print(f"--- Pipeline Complete: {len(self.manifest)} images processed ---")

    def _fetch_metadata(self, lat, lon, buffer):
        bbox = f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer}"
        fields = "id,thumb_original_url,computed_geometry,computed_compass_angle,camera_parameters,captured_at"
        url = f"https://graph.mapillary.com/images?bbox={bbox}&fields={fields}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                raw_data = response.json().get("data", [])
                self.manifest = [self._format_entry(img) for img in raw_data]
            else:
                print(f"[ERROR] API {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[CRITICAL] Fetch failed: {e}")

    def _format_entry(self, img):
        """Packs metadata into a structured format with labeled camera parameters."""
        params = img.get("camera_parameters", [])
        camera_data = {
            "raw_array": params,
            "description": "Spherical/No Distortion" if not params else "Perspective/Fisheye"
        }

        if len(params) >= 3:
            camera_data.update({
                "focal_length": params[0],
                "k1_radial_distortion": params[1],
                "k2_radial_distortion": params[2]
            })

        return {
            "image_id": img['id'],
            "url": img.get('thumb_original_url'),
            "captured_at": img.get('captured_at'),
            "pose": {
                "lat": img['computed_geometry']['coordinates'][1],
                "lon": img['computed_geometry']['coordinates'][0],
                "heading": img.get('computed_compass_angle')
            },
            "camera": camera_data
        }

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=4)
        print(f"Manifest saved to {self.manifest_path}")

    def _download_images(self):
        print(f"Downloading {len(self.manifest)} images to '{self.assets_dir}'...")
        for img in self.manifest:
            file_path = os.path.join(self.assets_dir, f"{img['image_id']}.jpg")
            
            if os.path.exists(file_path):
                continue

            try:
                response = requests.get(img['url'], stream=True)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
            except Exception as e:
                print(f"Failed to download {img['image_id']}: {e}")

if __name__ == "__main__":
    # Pierpont Commons
    processor = MapillaryIngestor()
    processor.run_pipeline(42.2912, -83.7175)