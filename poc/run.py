import os
import sys
import time

import poc.download_data as download_data
import poc.build_polygon as build_polygon
import poc.index_images as index_images
import poc.match_index as match_index

# --- Configuration ---
TARGET_LAT = 42.2912
TARGET_LON = -83.7175
BUFFER_DEG = 0.05
DATA_DIR = "data"
OSM_FILE = os.path.join(DATA_DIR, "osm_buildings.json")
MAPILLARY_FILE = os.path.join(DATA_DIR, "mapillary_images.json")

def step_1_download():
    print("\n[STEP 1/5] Fetching Metadata...")
    ingestor = download_data.Ingestor(base_dir=DATA_DIR)
    ingestor.manifest_path = MAPILLARY_FILE
    ingestor.osm_path = OSM_FILE
    ingestor.run_pipeline(TARGET_LAT, TARGET_LON, buffer=BUFFER_DEG)

def step_2_build_buildings():
    print("\n[STEP 2/5] Indexing Buildings...")
    build_polygon.INPUT_OSM_PATH = OSM_FILE
    build_polygon.OUTPUT_DIR = os.path.join(DATA_DIR, "res6_polygon")
    build_polygon.main()

def step_3_build_images():
    print("\n[STEP 3/5] Indexing Images...")
    index_images.INPUT_MANIFEST_PATH = MAPILLARY_FILE
    index_images.OUTPUT_DIR = os.path.join(DATA_DIR, "res6_images")
    index_images.main()

def step_4_match():
    print("\n[STEP 4/5] Matching Images to Buildings...")
    match_index.IMAGE_INDEX_PATH = os.path.join(DATA_DIR, "res6_images")
    match_index.BUILDING_INDEX_PATH = os.path.join(DATA_DIR, "res6_polygon")
    match_index.OSM_SOURCE_PATH = OSM_FILE
    match_index.OUTPUT_DIR = os.path.join(DATA_DIR, "final_matches")
    try:
        match_index.main()
    except SystemExit:
        pass

def step_5_download_assets():
    """Lazily downloads only the matched images."""
    print("\n[STEP 5/5] Hydrating Assets...")
    ingestor = download_data.Ingestor(base_dir=DATA_DIR)
    matches_path = os.path.join(DATA_DIR, "final_matches")
    ingestor.download_matches(matches_path)

def main():
    start_time = time.time()
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    step_1_download()
    step_2_build_buildings()
    step_3_build_images()
    step_4_match()
    step_5_download_assets()
    
    print(f"\n--- Pipeline Finished in {time.time() - start_time:.2f}s ---")
    os._exit(0)

if __name__ == "__main__":
    main()