import duckdb
import h3

def query_specific_location(lat, lon):
    # 1. Calculate the keys
    # These serve as our surgical entry points into the Data Lake
    target_res6 = h3.latlng_to_cell(lat, lon, 6)
    target_res10 = h3.latlng_to_cell(lat, lon, 10)
    
    print(f"--- Search Parameters ---")
    print(f"Target Lat/Lon: ({lat}, {lon})")
    print(f"Target H3 Index (Res 10): {target_res10}")
    print(f"Parent Partition (Res 6): {target_res6}")
    print(f"--------------------------\n")

    # 2. Run SQL Query directly on the files
    # DuckDB uses the path to skip unnecessary folders (Partition Pruning)
    query = f"""
        SELECT 
            h3_index, 
            h3_res6, 
            image_id, 
            assigned_building_id, 
            distance_meters,
            pose
        FROM 'data/final_matches/*/*.parquet' 
        WHERE h3_res6 = '{target_res6}' 
          AND h3_index = '{target_res10}'
    """
    
    try:
        # Executes the query and returns only the matching rows as a DataFrame
        result_df = duckdb.query(query).df()
        
        if result_df.empty:
            print("Result: No matches found for this specific H3 cell.")
        else:
            print("Result: Match Found!")
            # Using to_string() ensures the full H3 strings are visible without truncation
            print(result_df.to_string(index=False))
            result_df.to_csv("test.csv", index=False)
            
    except Exception as e:
        print(f"Error querying Data Lake: {e}")

# Test with Ann Arbor coordinates
query_specific_location(42.2912, -83.7175)