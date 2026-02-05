import pandas as pd
import folium
import math
import os
import json

# --- CONFIG ---
MATCHES_PATH = "data/final_matches"
OSM_PATH = "data/osm_buildings.json"
OUTPUT_HTML = "fov_debug_map.html"

# SETTINGS TO VISUALIZE (The "Permissible Range")
VISUAL_FOV = 180   # Wide angle (matches your production setting)
VISUAL_DIST = 60   # Distance limit (matches your production setting)

def get_destination_point(lat, lon, bearing, distance_m):
    R = 6378137 # Earth radius
    brng = math.radians(bearing)
    d = distance_m
    
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
                      math.cos(lat1)*math.sin(d/R)*math.cos(brng))
    
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                             math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    
    return math.degrees(lat2), math.degrees(lon2)

def create_fov_polygon(lat, lon, heading, fov, dist):
    points = [(lat, lon)] 
    start_angle = heading - (fov / 2)
    num_steps = int(fov / 10)
    step = fov / num_steps
    
    for i in range(num_steps + 1):
        angle = start_angle + (i * step)
        p_lat, p_lon = get_destination_point(lat, lon, angle, dist)
        points.append((p_lat, p_lon))
        
    points.append((lat, lon))
    return points

def main():
    print(f"Generating FOV Visualizer (FOV: {VISUAL_FOV}°, Dist: {VISUAL_DIST}m)...")
    
    # 1. Load Matches
    if not os.path.exists(MATCHES_PATH):
        print(f"Error: {MATCHES_PATH} not found.")
        return

    try:
        # Handle Hive Partitions (folders) or Single File
        if os.path.isdir(MATCHES_PATH):
            # Read first available partition
            parts = [p for p in os.listdir(MATCHES_PATH) if "h3_res6=" in p]
            if parts:
                df = pd.read_parquet(os.path.join(MATCHES_PATH, parts[0]))
            else:
                df = pd.read_parquet(MATCHES_PATH)
        else:
            df = pd.read_parquet(MATCHES_PATH)
    except Exception as e:
        print(f"Error loading matches: {e}")
        return

    if df.empty:
        print("No matches found.")
        return

    # 2. Pick Target Image
    target_id = 38663374399240
    if 'image_id' in df.columns and (df['image_id'] == target_id).any():
        print(f"Found specific image: {target_id}")
        row = df[df['image_id'] == target_id].iloc[0]
    else:
        # Fallback: Pick a row that actually matched a building
        print(f"Target image {target_id} not found (maybe filtered out?). Picking the first available match.")
        row = df.iloc[0]

    # Extract Pose
    pose = row.get('pose')
    if isinstance(pose, dict):
        lat, lon, heading = pose.get('lat'), pose.get('lon'), pose.get('heading')
    else:
        lat = row.get('lat') or row.get('pose.lat')
        lon = row.get('lon') or row.get('pose.lon')
        heading = row.get('heading') or row.get('pose.heading')

    if pd.isna(heading):
        print("Selected image has NO HEADING info. Cannot visualize FOV.")
        return
        
    print(f"Visualizing Image {row['image_id']} at ({lat:.5f}, {lon:.5f}) Heading {heading:.0f}°")

    # 3. Draw Map
    m = folium.Map(location=[lat, lon], zoom_start=20, tiles="CartoDB positron")

    # Cone
    fov_poly = create_fov_polygon(lat, lon, heading, VISUAL_FOV, VISUAL_DIST)
    folium.Polygon(
        locations=fov_poly,
        color="#FFA500", weight=2, fill=True, fill_color="#FFA500", fill_opacity=0.2,
        popup=f"FOV: {VISUAL_FOV}° | Dist: {VISUAL_DIST}m"
    ).add_to(m)

    # Heading Arrow
    arrow_end = get_destination_point(lat, lon, heading, VISUAL_DIST * 0.5)
    folium.PolyLine(locations=[(lat, lon), arrow_end], color="black", weight=3, opacity=0.8).add_to(m)

    # Camera Dot
    folium.CircleMarker(location=[lat, lon], radius=6, color="blue", fill=True, fill_color="blue", fill_opacity=1.0).add_to(m)

    # 4. Draw Buildings (FIXED JSON LOADING)
    if os.path.exists(OSM_PATH):
        with open(OSM_PATH, 'r') as f:
            data = json.load(f)
        
        # [FIX] Handle list vs dict format
        elements = data if isinstance(data, list) else data.get("elements", [])
        
        nodes = {el['id']: (el['lat'], el['lon']) for el in elements if el.get('type') == 'node'}
        
        for el in elements:
            if el.get('type') == 'way' and 'nodes' in el:
                coords = [nodes[n] for n in el['nodes'] if n in nodes]
                if len(coords) < 3: continue
                
                bid = el['id']
                is_match = (bid == row['assigned_building_id'])
                color = "#00FF00" if is_match else "#FF3333"
                opacity = 0.6 if is_match else 0.2
                
                folium.Polygon(
                    locations=coords, color=color, weight=2, fill=True, fill_color=color, fill_opacity=opacity,
                    popup=f"Building ID: {bid}"
                ).add_to(m)

    m.save(OUTPUT_HTML)
    print(f"Map saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()