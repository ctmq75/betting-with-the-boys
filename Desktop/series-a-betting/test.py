import pandas as pd
import os

# Test the exact path the script is looking for
csv_path = 'data/upcoming_matches.csv'
print(f"Looking for: {csv_path}")
print(f"File exists: {os.path.exists(csv_path)}")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    print("First match:", df.iloc[0]['home_team'], 'vs', df.iloc[0]['away_team'])
else:
    print("FILE NOT FOUND")
    print("Current directory:", os.getcwd())
    print("Files in current dir:", os.listdir('.'))