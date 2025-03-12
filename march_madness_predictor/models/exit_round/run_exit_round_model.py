import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add parent directory to path to allow importing model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exit_round.exit_round_model import TournamentExitPredictor

if __name__ == "__main__":
    print(f"Running Exit Round Prediction Model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up directories
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    DATA_DIR = os.path.join(BASE_DIR, "susan_kenpom")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "march_madness_predictor/models/exit_round/model")
    
    # Ensure model directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model save directory: {MODEL_SAVE_DIR}")
    
    # Create predictor
    predictor = TournamentExitPredictor(model_save_path=MODEL_SAVE_DIR)
    
    # Path to current season data
    current_season_file = os.path.join(DATA_DIR, "summary25.csv")
    
    if not os.path.exists(current_season_file):
        print(f"ERROR: Current season file not found at {current_season_file}")
        sys.exit(1)
    
    # Run full pipeline
    print("\n" + "="*80)
    print("Running exit round prediction pipeline...")
    print("="*80)
    
    try:
        # Load historical data, analyze seed performance, train model, and predict for current season
        predictions = predictor.run_full_pipeline(
            historical_data_dir=DATA_DIR,
            current_data_path=current_season_file
        )
        
        # Check if we got predictions back
        if predictions is None or len(predictions) == 0:
            print("WARNING: No predictions were generated. Using direct seed-based estimation.")
            # Load current season data
            try:
                current_data = pd.read_csv(current_season_file)
                # Try direct seed estimation
                predictions = predictor.estimate_seeds_for_2025(current_data)
            except Exception as e:
                print(f"ERROR: Failed to perform direct seed estimation: {e}")
                traceback.print_exc()
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*80)
    print("Exit Round Prediction Summary")
    print("="*80)
    
    # Count tournament teams (those with seed assignments)
    if predictions is not None:
        tournament_count = predictions['Seed'].notna().sum()
        print(f"Total teams predicted to make tournament: {tournament_count}")
        
        # Exit round distribution
        if 'PredictedExitRoundInt' in predictions.columns:
            # Only include non-null values
            exit_rounds = predictions.loc[predictions['PredictedExitRoundInt'].notna(), 'PredictedExitRoundInt'].value_counts().sort_index()
            print("\nPredicted Exit Round Distribution:")
            for exit_round, count in exit_rounds.items():
                if pd.notna(exit_round):
                    round_name = predictor.exit_round_mapping.get(int(exit_round), f"Unknown ({exit_round})")
                    print(f"  {round_name}: {count} teams")
        
        # Display top championship contenders
        if 'ChampionshipPct' in predictions.columns:
            print("\nTop 10 Championship Contenders:")
            top_10 = predictions.sort_values('ChampionshipPct', ascending=False).head(10)
            for i, (_, team) in enumerate(top_10.iterrows(), 1):
                seed_str = f"(Seed {int(team['Seed'])})" if pd.notna(team['Seed']) else "(Not in Tournament)"
                champ_pct = team['ChampionshipPct'] if pd.notna(team['ChampionshipPct']) else 0.0
                exit_round = team['PredictedExit'] if pd.notna(team['PredictedExit']) else "N/A"
                print(f"{i}. {team['TeamName']} {seed_str}: {champ_pct:.1f}% championship, Predicted: {exit_round}")
        
        # Display top Final Four contenders
        if 'FinalFourPct' in predictions.columns:
            print("\nTop 10 Final Four Contenders:")
            top_10_ff = predictions.sort_values('FinalFourPct', ascending=False).head(10)
            for i, (_, team) in enumerate(top_10_ff.iterrows(), 1):
                seed_str = f"(Seed {int(team['Seed'])})" if pd.notna(team['Seed']) else "(Not in Tournament)"
                ff_pct = team['FinalFourPct'] if pd.notna(team['FinalFourPct']) else 0.0
                print(f"{i}. {team['TeamName']} {seed_str}: {ff_pct:.1f}% Final Four")
        
        # Save predictions to CSV
        try:
            predictions.to_csv(os.path.join(MODEL_SAVE_DIR, 'exit_round_predictions.csv'), index=False)
            # Save tournament teams only
            tournament_teams = predictions[predictions['Seed'].notna()].copy()
            tournament_teams.to_csv(os.path.join(MODEL_SAVE_DIR, 'tournament_teams_predictions.csv'), index=False)
            
            print("\nPredictions saved to:")
            print(f"- {os.path.join(MODEL_SAVE_DIR, 'exit_round_predictions.csv')}")
            print(f"- {os.path.join(MODEL_SAVE_DIR, 'tournament_teams_predictions.csv')}")
        except Exception as e:
            print(f"WARNING: Could not save prediction files: {e}")
    else:
        print("ERROR: No predictions available to display")
    
    print("\nExit Round Prediction Model completed successfully!") 