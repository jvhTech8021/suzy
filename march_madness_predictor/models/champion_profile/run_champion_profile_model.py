import os
import sys
import datetime

# Add parent directory to path to import model
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from champion_profile.champion_profile_model import ChampionProfilePredictor

# Import directly from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from champion_profile_model import ChampionProfilePredictor

def main():
    """Run the champion profile prediction model"""
    print(f"=== Champion Profile Analysis - Run Date: {datetime.datetime.now()} ===")
    
    # Data and model paths
    # Get absolute path to workspace root directory
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(workspace_root, "susan_kenpom/summary25.csv")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Run the pipeline
    print(f"Running champion profile analysis with 2025 KenPom data from {data_path}")
    predictor = ChampionProfilePredictor(model_save_path=model_dir)
    predictions = predictor.run_full_pipeline(data_path)
    
    # Run tournament success level analysis
    print("\nRunning tournament success level analysis...")
    historical_data_dir = os.path.join(workspace_root, "susan_kenpom")
    tournament_results = predictor.analyze_tournament_success_levels(historical_data_dir, predictions)
    
    # Display summary
    print("\n" + "="*80)
    print("CHAMPION PROFILE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total teams analyzed: {len(predictions)}")
    print(f"Top 3 teams by champion profile similarity:")
    for i, (_, team) in enumerate(predictions.head(3).iterrows(), 1):
        print(f"{i}. {team['TeamName']}: {team['SimilarityPct']:.1f}% similarity")
    
    print("\nResults saved to:")
    print(f" - {os.path.join(model_dir, 'top30_champion_resemblers.csv')}")
    print(f" - {os.path.join(model_dir, 'all_teams_champion_profile.csv')}")
    print(f" - {os.path.join(model_dir, 'champion_profile_comparison.png')}")
    print(f" - {os.path.join(model_dir, 'top20_similarity.png')}")
    print(f" - {os.path.join(model_dir, 'championship_probabilities.png')}")
    print(f" - {os.path.join(model_dir, 'similarity_bracket.json')}")
    print(f" - {os.path.join(model_dir, 'similarity_bracket.txt')}")
    
    # Add summaries of tournament round analysis
    print("\nTournament success level analysis results:")
    for round_num, round_data in tournament_results.items():
        round_name = round_data['RoundName']
        team_count = round_data['TeamCount']
        print(f" - {round_name}: Analyzed {team_count} historical teams")
        print(f"   Saved to: {os.path.join(model_dir, round_name.lower().replace(' ', '_') + '_analysis.json')}")
    
    print("="*80)
    print("Champion Profile Analysis completed successfully!\n")

if __name__ == "__main__":
    main() 