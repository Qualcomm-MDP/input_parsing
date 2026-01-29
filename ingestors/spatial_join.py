import json
import geopandas as gpd
from shapely.geometry import Point, Polygon

def spatial_join(manifest_path="data/manifest.json", osm_path="data/osm_buildings.json"):
    # 1. Load data from the ingestor outputs
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    with open(osm_path, "r") as f:
        osm_elements = json.load(f)

    # 2. Process Mapillary imagery metadata
    img_list = []
    for img in manifest:
        img_list.append({
            "image_id": img['image_id'],
            "sequence_id": img['sequence_id'],
            "lat": img['pose']['lat'],
            "lon": img['pose']['lon'],
            "heading": img['pose']['heading'],
            "geometry": Point(img['pose']['lon'], img['pose']['lat'])
        })
    images_gdf = gpd.GeoDataFrame(img_list, crs="EPSG:4326")

    # 3. Process OSM building geometries
    nodes = {el['id']: (el['lon'], el['lat']) for el in osm_elements if el['type'] == 'node'}
    
    building_list = []
    for el in osm_elements:
        if el['type'] == 'way' and 'nodes' in el:
            coords = [nodes[node_id] for node_id in el['nodes'] if node_id in nodes]
            if len(coords) >= 3:
                building_list.append({
                    "osm_id": el['id'],
                    "building_name": el.get('tags', {}).get('name', 'Unknown'),
                    "geometry": Polygon(coords)
                })
    buildings_gdf = gpd.GeoDataFrame(building_list, crs="EPSG:4326")

    # 4. Perform Spatial Join (Nearest Building)
    images_proj = images_gdf.to_crs(epsg=3857)
    buildings_proj = buildings_gdf.to_crs(epsg=3857)
    
    # Calculate distance in meters between camera pose and nearest building
    result = gpd.sjoin_nearest(images_proj, buildings_proj, distance_col="distance_m", how="left")

    # 5. Reorder columns for human-readable output
    # This specific order ensures building name and distance appear first in the JSON properties
    ordered_cols = [
        'building_name', 
        'distance_m', 
        'image_id', 
        'sequence_id', 
        'heading', 
        'lat', 
        'lon', 
        'osm_id', 
        'geometry'
    ]
    
    # Final output conversion to GeoJSON
    final_gdf = result[ordered_cols].to_crs(epsg=4326)
    
    output_path = "data/spatial_join.json"
    final_gdf.to_file(output_path, driver='GeoJSON')
    print(f"Readable join results saved to {output_path}")

if __name__ == "__main__":
    spatial_join()