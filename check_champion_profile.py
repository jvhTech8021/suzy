#!/usr/bin/env python3
import os
import json
import pandas as pd
from pathlib import Path

def main():
    """Check the champion profile data directly"""
    print("=== CHECKING CHAMPION PROFILE DATA ===")
    
    # Get the absolute path to the champion profile model directory
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "march_madness_predictor" / "models" / "champion_profile" / "model"
    
    # Print debug information
    print(f"Looking for champion profile data in: {model_dir}")
    json_path = model_dir / "champion_profile.json"
    csv_path = model_dir / "all_teams_champion_profile.csv"
    
    print(f"JSON file exists: {json_path.exists()}, size: {json_path.stat().st_size if json_path.exists() else 0} bytes")
    print(f"CSV file exists: {csv_path.exists()}, size: {csv_path.stat().st_size if csv_path.exists() else 0} bytes")
    
    # Try loading the JSON file
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                file_content = f.read()
                print(f"Raw JSON content ({len(file_content)} bytes):")
                print(repr(file_content))
                
                # Clean the content if needed
                if file_content.endswith('%'):
                    file_content = file_content[:-1]
                    print("Removed trailing '%' character")
                
                # Try to parse the JSON
                champion_profile = json.loads(file_content)
                print(f"Successfully parsed JSON: {champion_profile}")
                
                # Write back the clean JSON
                with open(json_path, 'w') as f:
                    json.dump(champion_profile, f, indent=4)
                print(f"Wrote clean JSON back to {json_path}")
        except Exception as e:
            print(f"ERROR loading JSON file: {type(e).__name__}: {str(e)}")
    
    # Try loading the CSV file
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print("Columns:", df.columns.tolist())
            print("First row:", df.iloc[0].to_dict())
        except Exception as e:
            print(f"ERROR loading CSV file: {type(e).__name__}: {str(e)}")
    
    print("\n=== CHECK COMPLETE ===")

if __name__ == "__main__":
    main() 