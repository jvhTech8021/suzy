import os
import pandas as pd
import json
import sys

# Add the project directory to the path
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

# Import the DataLoader
from utils.data_loader import DataLoader

def main():
    """Main function to debug data loading"""
    print("=== DATA LOADING DEBUG ===")
    
    # Initialize the DataLoader
    data_loader = DataLoader()
    print(f"Base path: {data_loader.base_path}")
    print(f"KenPom data dir: {data_loader.kenpom_data_dir}")
    print(f"Champion profile dir: {data_loader.champion_profile_dir}")
    print(f"Exit round dir: {data_loader.exit_round_dir}")
    
    # Check if directories exist
    print("\nChecking directories:")
    print(f"KenPom data dir exists: {os.path.exists(data_loader.kenpom_data_dir)}")
    print(f"Champion profile dir exists: {os.path.exists(data_loader.champion_profile_dir)}")
    print(f"Exit round dir exists: {os.path.exists(data_loader.exit_round_dir)}")
    
    # Check for key files
    print("\nChecking key files:")
    
    # Current season data
    current_season_file = os.path.join(data_loader.kenpom_data_dir, "summary25.csv")
    print(f"Current season file exists: {os.path.exists(current_season_file)}")
    
    # Champion profile files
    champion_profile_json = os.path.join(data_loader.champion_profile_dir, "champion_profile.json")
    all_teams_profile_csv = os.path.join(data_loader.champion_profile_dir, "all_teams_champion_profile.csv")
    print(f"Champion profile JSON exists: {os.path.exists(champion_profile_json)}")
    print(f"All teams champion profile CSV exists: {os.path.exists(all_teams_profile_csv)}")
    
    # Exit round files
    exit_round_predictions_csv = os.path.join(data_loader.exit_round_dir, "exit_round_predictions.csv")
    tournament_teams_csv = os.path.join(data_loader.exit_round_dir, "tournament_teams_predictions.csv")
    print(f"Exit round predictions CSV exists: {os.path.exists(exit_round_predictions_csv)}")
    print(f"Tournament teams CSV exists: {os.path.exists(tournament_teams_csv)}")
    
    # Try loading data
    print("\nTrying to load data:")
    
    try:
        current_data = data_loader.get_current_season_data()
        print(f"Current season data loaded: {len(current_data)} teams")
    except Exception as e:
        print(f"Error loading current season data: {e}")
    
    try:
        champion_profile = data_loader.get_champion_profile()
        print(f"Champion profile loaded: {champion_profile}")
    except Exception as e:
        print(f"Error loading champion profile: {e}")
    
    try:
        champion_predictions = data_loader.get_champion_profile_predictions()
        print(f"Champion profile predictions loaded: {len(champion_predictions)} teams")
        print(f"Top 3 teams: {champion_predictions['TeamName'].head(3).tolist()}")
    except Exception as e:
        print(f"Error loading champion profile predictions: {e}")
    
    try:
        exit_predictions = data_loader.get_exit_round_predictions()
        print(f"Exit round predictions loaded: {len(exit_predictions)} teams")
        print(f"Top 3 teams: {exit_predictions['TeamName'].head(3).tolist()}")
    except Exception as e:
        print(f"Error loading exit round predictions: {e}")
    
    try:
        combined_predictions = data_loader.get_combined_predictions()
        if combined_predictions is not None:
            print(f"Combined predictions loaded: {len(combined_predictions)} teams")
            print(f"Top 3 teams: {combined_predictions['TeamName'].head(3).tolist()}")
        else:
            print("Combined predictions returned None")
    except Exception as e:
        print(f"Error loading combined predictions: {e}")
    
    print("\n=== END DEBUG ===")

if __name__ == "__main__":
    main() 