import pandas as pd
df = pd.read_parquet("data/final_matches") # Load your results

print(f"Total Rows (Matches): {len(df)}")           # Should be 171
print(f"Unique Images: {df['image_id'].nunique()}") # Should be 29