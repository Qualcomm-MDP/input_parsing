import json
import h3
import pandas as pd
import os

# --- Configuration ---
INPUT_MANIFEST_PATH = "data/mapillary_images.json"
OUTPUT_DIR = "data/res6_images"
H3_RESOLUTION = 10 

def main():
    print(f"Loading image manifest from {INPUT_MANIFEST_PATH}...")
    
    if not os.path.exists(INPUT_MANIFEST_PATH):
        print("Error: manifest.json not found. Run your pipeline.py first to download metadata.")
        return

    with open(INPUT_MANIFEST_PATH, 'r') as f:
        images = json.load(f)

    print(f"Found {len(images)} images. Indexing...")

    index_data = []
    for img in images:
        lat = img['pose']['lat']
        lon = img['pose']['lon']
        
        h3_index = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        
        # [FIX] Directory is now 'data/assets' to match downloader
        # [NEW] Storing 'url' so we can lazy-load later
        index_data.append({
            "h3_index": h3_index,
            "image_id": img['image_id'],
            "sequence_id": img.get('sequence_id'),
            "lat": lat,
            "lon": lon,
            "heading": img['pose']['heading'],
            "captured_at": img.get('captured_at'),
            "url": img.get('url'),
            "camera_type": img.get('camera_type', 'perspective'),
            "image_path": f"data/assets/{img.get('sequence_id')}/{img['image_id']}.jpg"
        })

    if not index_data:
        print("No data extracted.")
        return

    # Create DataFrame
    df = pd.DataFrame(index_data)

    # Partition and Save
    print("Calculating partitions...")
    df['h3_res6'] = df['h3_index'].apply(lambda x: h3.cell_to_parent(x, 6))

    print(f"Saving to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df.to_parquet(
        OUTPUT_DIR,
        partition_cols=['h3_res6'],
        compression='snappy',
        index=False
    )
    print("Done. Image index created.")

if __name__ == "__main__":
    main()