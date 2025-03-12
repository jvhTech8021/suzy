import os
import sys
import json
import pandas as pd

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the DataLoader
from march_madness_predictor.utils.data_loader import DataLoader

def main():
    """Debug champion profile data loading"""
    print("=== CHAMPION PROFILE DEBUG ===")
    
    # Initialize the DataLoader
    data_loader = DataLoader()
    print(f"Base path: {data_loader.base_path}")
    print(f"Champion profile dir: {data_loader.champion_profile_dir}")
    
    # Check if directory exists
    print(f"Champion profile dir exists: {os.path.exists(data_loader.champion_profile_dir)}")
    
    # Check for key files
    champion_profile_json = os.path.join(data_loader.champion_profile_dir, "champion_profile.json")
    all_teams_profile_csv = os.path.join(data_loader.champion_profile_dir, "all_teams_champion_profile.csv")
    
    print(f"champion_profile.json exists: {os.path.exists(champion_profile_json)}")
    print(f"all_teams_champion_profile.csv exists: {os.path.exists(all_teams_profile_csv)}")
    
    # Read file contents directly
    try:
        print("\nReading champion_profile.json directly:")
        with open(champion_profile_json, 'r') as f:
            data = json.load(f)
            print(f"Content: {data}")
    except Exception as e:
        print(f"Error reading champion_profile.json: {e}")
    
    try:
        print("\nReading all_teams_champion_profile.csv directly (first 5 rows):")
        df = pd.read_csv(all_teams_profile_csv)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 5 rows:\n{df.head(5)}")
    except Exception as e:
        print(f"Error reading all_teams_champion_profile.csv: {e}")
    
    # Try using the DataLoader methods
    print("\nTrying to load data using DataLoader methods:")
    try:
        champion_profile = data_loader.get_champion_profile()
        print(f"Successfully loaded champion_profile: {champion_profile}")
    except Exception as e:
        print(f"Error loading champion_profile: {e}")
    
    try:
        predictions = data_loader.get_champion_profile_predictions()
        print(f"Successfully loaded champion_profile_predictions with shape: {predictions.shape}")
        print(f"First 3 teams: {predictions['TeamName'].head(3).tolist()}")
    except Exception as e:
        print(f"Error loading champion_profile_predictions: {e}")
    
    print("\n=== END DEBUG ===")

if __name__ == "__main__":
    main() 