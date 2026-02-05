import pandas as pd
import json
import h3
import os
import folium
import math

# --- Configuration ---
MATCHES_PATH = "data/final_matches"
OSM_PATH = "data/osm_buildings.json"
OUTPUT_HTML = "matches_map.html"

def load_building_geometries(path):
    if not os.path.exists(path):
        print(f"Warning: OSM file not found at {path}")
        return {}
    
    with open(path, 'r') as f:
        data = json.load(f)

    elements = data if isinstance(data, list) else data.get("elements", [])
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in elements if el.get("type") == "node"}
    geoms = {}

    for el in elements:
        if el.get("type") == "way" and "nodes" in el:
            coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
            if len(coords) >= 3:
                geoms[el["id"]] = coords 
    return geoms

def visualize_area(lat, lon):
    print(f"Generating map for area around ({lat}, {lon})...")
    
    # 1. Determine Partition (Res 6) to speed up loading
    target_res6 = h3.latlng_to_cell(lat, lon, 6)
    print(f"Target Partition: {target_res6}")

    if not os.path.exists(MATCHES_PATH):
        print(f"Error: Matches file not found at {MATCHES_PATH}. Run step_4_match first.")
        return

    # 2. Load Matches (Robust Partition Read)
    # Pandas read_parquet handles Hive partitions (folders like h3_res6=...) automatically
    try:
        # We explicitly look for the folder if strict filtering fails
        partition_path = os.path.join(MATCHES_PATH, f"h3_res6={target_res6}")
        if os.path.exists(partition_path):
            print(f"Loading specific partition: {partition_path}")
            df = pd.read_parquet(partition_path)
        else:
            print("Partition not found on disk. Trying global load...")
            df = pd.read_parquet(MATCHES_PATH)
            # Filter manually if column exists (it might not if hive-loaded without index)
            if 'h3_res6' in df.columns:
                df = df[df['h3_res6'] == target_res6]
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return

    if df.empty:
        print("No matches found in this partition.")
        return

    # 3. Load Building Shapes
    building_shapes = load_building_geometries(OSM_PATH)

    # 4. Initialize Map
    m = folium.Map(location=[lat, lon], zoom_start=18, tiles="CartoDB positron")

    # 5. Draw Data
    count = 0
    
    # We iterate rows. Since we have One-to-Many matches, 
    # one image_id might appear multiple times (pointing to different buildings).
    for _, row in df.iterrows():
        # Handle Pose (sometimes stored as struct/dict, sometimes flat columns)
        pose = row.get('pose')
        if isinstance(pose, dict):
            img_lat = pose.get('lat')
            img_lon = pose.get('lon')
            heading = pose.get('heading')
        else:
            img_lat = row.get('lat') or row.get('pose.lat')
            img_lon = row.get('lon') or row.get('pose.lon')
            heading = row.get('heading') or row.get('pose.heading')
        
        # Get Target (Wall) Location
        target_lat = row.get('target_lat')
        target_lon = row.get('target_lon')

        if pd.isna(img_lat) or pd.isna(img_lon) or pd.isna(target_lat) or pd.isna(target_lon):
            continue

        # --- LAZY LOADING LOGIC ---
        # 1. Try Local File
        local_path = os.path.abspath(row['image_path'])
        
        if os.path.exists(local_path):
            img_src = f"file://{local_path}"
            source_label = "Local Disk"
        else:
            # 2. Fallback to Web URL
            img_src = row.get('url', '')
            source_label = "Web URL (Not downloaded)"

        # Draw Image Marker (Camera Position)
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
            <b>Image ID:</b> {row['image_id']}<br>
            <b>Building ID:</b> {row['assigned_building_id']}<br>
            <b>Status:</b> {source_label}<br>
            <b>Dist:</b> {row['distance_meters']}m<br>
            <hr>
            <img src='{img_src}' width='100%' style='border-radius: 4px; min-height: 100px; background: #eee;'>
        </div>
        """
        
        folium.CircleMarker(
            location=[img_lat, img_lon],
            radius=4,
            color="#3388ff",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

        # Draw View Frustum / Connection Line
        # Green Line = Camera to Target Wall Contact Point
        folium.PolyLine(
            locations=[(img_lat, img_lon), (target_lat, target_lon)],
            color="green",
            weight=2,
            opacity=0.6,
            dash_array='5, 5'
        ).add_to(m)

        # Draw Building Polygon (Red)
        bid = row['assigned_building_id']
        if bid in building_shapes:
            shape = building_shapes[bid]
            folium.Polygon(
                locations=shape,
                color="#ff3333",
                weight=1,
                fill=True,
                fill_color="#ff3333",
                fill_opacity=0.1,
                popup=f"Building: {bid}"
            ).add_to(m)
            
        count += 1

    m.save(OUTPUT_HTML)
    print(f"Map saved to {OUTPUT_HTML}. Visualized {count} matches.")
    print("Open this file in your browser to view.")

if __name__ == "__main__":
    # Default to North Campus coordinates
    visualize_area(42.2912, -83.7175)