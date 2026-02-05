import json
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import os

# --- Configuration ---
INPUT_OSM_PATH = "data/osm_buildings.json"  # Was data/raw_osm_geometries.json
OUTPUT_DIR = "data/res6_polygon"
H3_RESOLUTION = 10        # Res 10 is ~15,000 m^2
BUFFER_K_RING = 1         # 1 ring = immediate neighbors

def load_osm_geometry(path):
    """
    Parses raw Overpass JSON into a GeoDataFrame.
    Handles both raw API response (dict) and pipeline output (list).
    """
    print(f"Loading OSM data from {path}...")
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return gpd.GeoDataFrame()

    with open(path, 'r') as f:
        data = json.load(f)

    # Handle List vs Dictionary input
    if isinstance(data, list):
        elements = data
    else:
        elements = data.get("elements", [])

    nodes = {}
    for el in elements:
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    building_records = []
    for el in elements:
        if el.get("type") == "way" and "nodes" in el:
            if "building" in el.get("tags", {}):
                coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
                if len(coords) >= 3:
                    poly = Polygon(coords)
                    building_records.append({
                        "osm_id": el["id"],
                        "name": el.get("tags", {}).get("name", "Unknown"),
                        "geometry": poly
                    })
    
    gdf = gpd.GeoDataFrame(building_records, crs="EPSG:4326")
    print(f"Parsed {len(gdf)} valid buildings.")
    return gdf

def get_h3_catchment(geometry, res, k_ring):
    """
    Returns H3 indices for building + surroundings.
    """
    geo_json = mapping(geometry)
    try:
        # H3 v4 function: polygon_to_cells
        filled_cells = h3.polygon_to_cells(geo_json, res)
    except Exception:
        filled_cells = set()

    if not filled_cells:
        centroid = geometry.centroid
        # H3 v4 function: latlng_to_cell
        center_cell = h3.latlng_to_cell(centroid.y, centroid.x, res)
        filled_cells = {center_cell}

    catchment = set()
    for cell in filled_cells:
        # H3 v4 function: grid_disk
        neighbors = h3.grid_disk(cell, k_ring)
        catchment.update(neighbors)
        
    return list(catchment)

def main():
    # 1. Load Data
    gdf_buildings = load_osm_geometry(INPUT_OSM_PATH)
    
    if gdf_buildings.empty:
        print("No buildings to process. Exiting.")
        return

    print("Generating H3 catchment areas...")
    
    # 2. Explode: Create one row per H3 index per Building
    index_data = []
    
    for _, row in gdf_buildings.iterrows():
        osm_id = row['osm_id']
        geom = row['geometry']
        
        cells = get_h3_catchment(geom, H3_RESOLUTION, BUFFER_K_RING)
        
        for cell in cells:
            index_data.append({
                "h3_index": cell,
                "osm_id": osm_id
            })

    if not index_data:
        print("No H3 indices generated. Check geometry or resolution.")
        return

    # 3. Create the Inverted Index DataFrame
    df_index = pd.DataFrame(index_data)
    
    # Group by H3 Index -> List of Buildings
    df_grouped = df_index.groupby("h3_index")['osm_id'].apply(list).reset_index()
    df_grouped.rename(columns={'osm_id': 'building_ids'}, inplace=True)

    # 4. PARTITIONING LOGIC (Fixed for H3 v4)
    print("Calculating parent partitions (Res 6)...")
    
    # FIXED: Use 'cell_to_parent' instead of 'h3_to_parent'
    df_grouped['h3_res6'] = df_grouped['h3_index'].apply(lambda x: h3.cell_to_parent(x, 6))

    # 5. Save with Partitioning
    print(f"Saving partitioned parquet to {OUTPUT_DIR}...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_grouped.to_parquet(
        OUTPUT_DIR,
        partition_cols=['h3_res6'],
        compression='snappy',
        index=False
    )
    
    print(f"Done. Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()