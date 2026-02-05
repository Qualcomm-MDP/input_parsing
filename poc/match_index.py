import pandas as pd
import json
import os
import math
import shutil
import h3
from shapely.geometry import Point, Polygon
import pyarrow.parquet as pq

# --- Configuration ---
IMAGE_INDEX_PATH = "data/res6_images"
BUILDING_INDEX_PATH = "data/res6_polygon"
OSM_SOURCE_PATH = "data/osm_buildings.json"
OUTPUT_DIR = "data/final_matches"

# --- 1. GEOMETRY HELPERS ---

def point_to_segment_distance_m(lat, lon, a_lat, a_lon, b_lat, b_lon):
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
        return math.hypot(px - ax, py - ay), a_lat, a_lon

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t)) 
    
    hit_lat = a_lat + t * (b_lat - a_lat)
    hit_lon = a_lon + t * (b_lon - a_lon)
    
    cx, cy = ax + t * abx, ay + t * aby
    dist_m = math.hypot(px - cx, py - cy)
    
    return dist_m, hit_lat, hit_lon

def get_best_match_geometry(lat, lon, polygon):
    if polygon.contains(Point(lon, lat)):
        return 0.0, lat, lon

    coords = list(polygon.exterior.coords)
    best_dist = float("inf")
    best_point = (None, None)
    
    for i in range(len(coords) - 1):
        d, t_lat, t_lon = point_to_segment_distance_m(lat, lon, coords[i][1], coords[i][0], coords[i+1][1], coords[i+1][0])
        if d < best_dist:
            best_dist = d
            best_point = (t_lat, t_lon)
            
    return best_dist, best_point[0], best_point[1]

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360

def is_within_fov(camera_heading, target_bearing, fov=90):
    diff = abs(camera_heading - target_bearing)
    diff = diff if diff <= 180 else 360 - diff
    return diff <= (fov / 2)

def load_building_geometries(path):
    if not os.path.exists(path): return {}
    with open(path, 'r') as f: data = json.load(f)
    elements = data if isinstance(data, list) else data.get("elements", [])
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el.get("type") == "node"}
    geoms = {}
    for el in elements:
        if el.get("type") == "way" and "nodes" in el:
            coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
            if len(coords) >= 3: geoms[el["id"]] = Polygon(coords)
    return geoms

# --- 2. PIPELINE EXECUTION ---

def main():
    print("--- Starting Optimized Matching (One-to-Many Mode) ---")
    
    if os.path.exists(OUTPUT_DIR): 
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(IMAGE_INDEX_PATH):
        print(f"Error: {IMAGE_INDEX_PATH} not found.")
        return

    # Identify partitions
    image_partitions = [d.split('=')[1] for d in os.listdir(IMAGE_INDEX_PATH) if 'h3_res6=' in d]
    
    print(f"Found {len(image_partitions)} partitions. Loading Geometry...")
    geom_lookup = load_building_geometries(OSM_SOURCE_PATH)
    total_matches = 0

    for part_idx, current_res6 in enumerate(image_partitions):
        print(f"[{part_idx+1}/{len(image_partitions)}] Processing Partition {current_res6}...")
        
        # A. Load Images
        try:
            df_images = pd.read_parquet(IMAGE_INDEX_PATH, filters=[('h3_res6', '==', current_res6)])
        except Exception:
            continue
        if df_images.empty: continue

        # B. Smart Neighbor Expansion (K-Ring)
        if 'h3_index' not in df_images.columns: df_images = df_images.reset_index()
        
        # Create neighbor keys (Image A -> [A, B, C, D...])
        df_images['match_h3_keys'] = df_images['h3_index'].apply(lambda x: list(h3.grid_disk(x, 1)))
        
        # Explode
        df_images_expanded = df_images.explode('match_h3_keys')
        required_building_cells = df_images_expanded['match_h3_keys'].unique()
        required_parents = set(h3.cell_to_parent(c, 6) for c in required_building_cells)
        
        # C. Load Buildings
        try:
            df_buildings = pd.read_parquet(
                BUILDING_INDEX_PATH, 
                filters=[('h3_res6', 'in', list(required_parents))]
            )
        except Exception:
            continue
        
        if df_buildings.empty: continue
        
        if 'h3_index' not in df_buildings.columns: df_buildings = df_buildings.reset_index()
        
        # D. The Spatial Join
        merged = pd.merge(
            df_images_expanded, 
            df_buildings, 
            left_on="match_h3_keys", 
            right_on="h3_index", 
            how="inner",
            suffixes=('', '_bldg')
        )
        
        if merged.empty: continue

        # E. Geometric Refinement (MULTI-MATCH ENABLED)
        merged = merged.explode('building_ids')
        merged.rename(columns={'building_ids': 'candidate_building_id'}, inplace=True)
        
        partition_matches = []
        grouped = merged.groupby("image_id")
        
        for image_id, group in grouped:
            meta = group.iloc[0]
            img_lat, img_lon = meta['lat'], meta['lon']

            cam_heading = meta.get('heading')
            if pd.isna(cam_heading): cam_heading = None
            
            # [FIX] Dedup Set per image
            seen_buildings = set()

            for _, row in group.iterrows():
                bid = row['candidate_building_id']
                if bid in seen_buildings: continue
                seen_buildings.add(bid)

                poly = geom_lookup.get(bid)
                if not poly: continue
                
                # 1. Distance Check
                dist, t_lat, t_lon = get_best_match_geometry(img_lat, img_lon, poly)
                
                # 2. Angle Check (With Dynamic FOV)
                is_facing = True
                if cam_heading is not None:
                    bearing = calculate_bearing(img_lat, img_lon, t_lat, t_lon)
                    # [FIX] RELAXED FOV: Using 200 as base fallback or dynamic
                    is_facing = is_within_fov(cam_heading, bearing, fov=180)

                if is_facing and dist < 60.0:
                    partition_matches.append({
                        "image_id": image_id,
                        "h3_index": meta['h3_index'],
                        "assigned_building_id": bid,
                        "distance_meters": round(dist, 2),
                        "target_lat": t_lat,
                        "target_lon": t_lon,
                        "image_path": meta['image_path'],
                        "url": meta.get('url'),
                        "pose": {"lat": img_lat, "lon": img_lon, "heading": cam_heading},
                        "captured_at": meta.get('captured_at')
                    })

        # F. Write Partition Result
        if partition_matches:
            df_final = pd.DataFrame(partition_matches)
            
            out_path = os.path.join(OUTPUT_DIR, f"h3_res6={current_res6}")
            os.makedirs(out_path, exist_ok=True)
            
            df_final.to_parquet(
                os.path.join(out_path, "matches.parquet"),
                compression='snappy',
                index=False
            )
            total_matches += len(df_final)
            print(f"  > Saved {len(df_final)} matches (One-to-Many).")

    print(f"--- Completed. Total Matches: {total_matches} ---")

if __name__ == "__main__":
    main()