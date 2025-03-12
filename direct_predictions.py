import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

def make_direct_predictions():
    """
    Make predictions directly using the trained model without relying on the model loading function
    """
    # Paths
    model_path = 'march_madness_predictor/models/exit_round/model/exit_round_model.keras'
    scaler_path = 'march_madness_predictor/models/exit_round/model/scaler.joblib'
    feature_cols_path = 'march_madness_predictor/models/exit_round/model/feature_cols.json'
    current_data_path = 'susan_kenpom/summary25.csv'
    output_path = 'march_madness_predictor/models/exit_round/model/neural_network_predictions.csv'
    
    # Check if all necessary files exist
    for path in [model_path, scaler_path, feature_cols_path, current_data_path]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            return False
    
    print("Loading trained model...")
    try:
        # Load model with explicit custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Load scaler
    print("Loading scaler...")
    scaler = joblib.load(scaler_path)
    
    # Load feature columns
    print("Loading feature columns...")
    with open(feature_cols_path, 'r') as f:
        feature_cols = json.load(f)
    
    # Load current season data
    print("Loading current season data...")
    current_data = pd.read_csv(current_data_path)
    
    # Clean data
    for col in current_data.columns:
        if current_data[col].dtype == 'object' and col != 'TeamName':
            current_data[col] = current_data[col].str.replace('"', '').astype(float)
        elif col == 'TeamName':
            current_data[col] = current_data[col].str.replace('"', '')
    
    # Estimate seeds (simpler version)
    print("Estimating seeds for teams...")
    # Sort by adjusted efficiency margin
    if 'AdjEM' in current_data.columns:
        ranked_teams = current_data.sort_values('AdjEM', ascending=False).reset_index(drop=True)
    else:
        ranked_teams = current_data.sort_values('RankAdjEM', ascending=True).reset_index(drop=True)
    
    # Add Seed column if it doesn't exist
    if 'Seed' not in current_data.columns:
        current_data['Seed'] = np.nan
    
    # Take top 64 teams
    tournament_teams = ranked_teams.iloc[:64].copy()
    
    # Assign seeds (1-16 for 4 regions)
    seeds_per_region = 16
    num_regions = 4
    
    for i in range(seeds_per_region):
        # For even rows, go left to right
        if i % 2 == 0:
            for j in range(num_regions):
                idx = i * num_regions + j
                if idx < len(tournament_teams):
                    tournament_teams.iloc[idx, tournament_teams.columns.get_loc('Seed')] = i + 1
        # For odd rows, go right to left
        else:
            for j in range(num_regions - 1, -1, -1):
                idx = i * num_regions + (num_regions - 1 - j)
                if idx < len(tournament_teams):
                    tournament_teams.iloc[idx, tournament_teams.columns.get_loc('Seed')] = i + 1
    
    # Update the original dataframe with the estimated tournament data
    for i, row in tournament_teams.iterrows():
        team_idx = current_data[current_data['TeamName'] == row['TeamName']].index
        if len(team_idx) > 0:
            current_data.loc[team_idx, 'Seed'] = row['Seed']
    
    # Get tournament teams for prediction
    print(f"Preparing features for {len(tournament_teams)} teams...")
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in tournament_teams.columns]
    if missing_cols:
        print(f"Warning: Missing feature columns: {missing_cols}")
        print("Adding missing columns with default values (0)")
        for col in missing_cols:
            tournament_teams[col] = 0
    
    # Get features for prediction
    X = tournament_teams[feature_cols].values
    
    # Handle any missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    print("Generating neural network predictions...")
    y_pred = model.predict(X_scaled)
    
    # Round predictions to nearest integer (representing exit round)
    rounded_preds = np.round(y_pred).astype(int).flatten()
    
    # Clip predictions to valid range (1-7)
    rounded_preds = np.clip(rounded_preds, 1, 7)
    
    # Store predictions
    tournament_teams['NNPredictedExitRound'] = rounded_preds
    
    # Map exit rounds to names
    exit_round_mapping = {
        0: 'Did Not Make Tournament',
        1: 'First Round',
        2: 'Second Round',
        3: 'Sweet 16',
        4: 'Elite 8',
        5: 'Final Four',
        6: 'Championship Game',
        7: 'National Champion'
    }
    
    tournament_teams['NNPredictedExit'] = tournament_teams['NNPredictedExitRound'].apply(
        lambda x: exit_round_mapping.get(int(x), f"Unknown ({x})")
    )
    
    # Calculate championship probabilities based on exit round
    print("Calculating championship and Final Four probabilities...")
    tournament_teams['NNChampionshipPct'] = 0.0
    tournament_teams['NNFinalFourPct'] = 0.0
    
    for i, row in tournament_teams.iterrows():
        if pd.notna(row['Seed']) and pd.notna(row['NNPredictedExitRound']):
            seed = int(row['Seed'])
            exit_round = int(row['NNPredictedExitRound'])
            
            # Championship probability
            if exit_round == 7:  # Predicted champion
                champ_pct = 90.0
            elif exit_round == 6:  # Predicted runner-up
                champ_pct = 40.0
            elif exit_round == 5:  # Predicted Final Four
                champ_pct = 20.0
            elif exit_round == 4:  # Predicted Elite Eight
                champ_pct = 10.0
            elif seed <= 4:  # Top 4 seeds
                champ_pct = 5.0
            elif seed <= 8:
                champ_pct = 1.0
            else:
                champ_pct = 0.5
                
            # Final Four probability
            if exit_round >= 5:  # Predicted Final Four or better
                ff_pct = 90.0
            elif exit_round == 4:  # Predicted Elite Eight
                ff_pct = 40.0
            elif exit_round == 3:  # Predicted Sweet Sixteen
                ff_pct = 15.0
            elif seed <= 4:  # Top 4 seeds
                ff_pct = 25.0 
            elif seed <= 8:
                ff_pct = 10.0
            else:
                ff_pct = 5.0
                
            tournament_teams.loc[i, 'NNChampionshipPct'] = champ_pct
            tournament_teams.loc[i, 'NNFinalFourPct'] = ff_pct
    
    # Save tournament teams predictions
    tournament_teams.to_csv(output_path, index=False)
    
    # Print summary
    print("\n================================================================================")
    print("Neural Network Prediction Summary")
    print("================================================================================")
    print(f"Total teams predicted to make tournament: {len(tournament_teams)}")
    
    # Print distribution of predicted exit rounds
    exit_round_counts = tournament_teams['NNPredictedExitRound'].value_counts().sort_index()
    print("\nPredicted Exit Round Distribution:")
    for exit_round, count in exit_round_counts.items():
        exit_name = exit_round_mapping.get(int(exit_round), f"Unknown ({exit_round})")
        print(f"  {exit_name}: {count} teams")
    
    # Print top 10 championship contenders
    print("\nTop 10 Championship Contenders:")
    top_champions = tournament_teams.sort_values('NNChampionshipPct', ascending=False).head(10)
    for i, (_, row) in enumerate(top_champions.iterrows(), 1):
        print(f"{i}. {row['TeamName']} (Seed {int(row['Seed'])}): {row['NNChampionshipPct']}% championship, Predicted: {row['NNPredictedExit']}")
    
    # Print top 10 Final Four contenders
    print("\nTop 10 Final Four Contenders:")
    top_ff = tournament_teams.sort_values('NNFinalFourPct', ascending=False).head(10)
    for i, (_, row) in enumerate(top_ff.iterrows(), 1):
        print(f"{i}. {row['TeamName']} (Seed {int(row['Seed'])}): {row['NNFinalFourPct']}% Final Four")
    
    print(f"\nNeural network predictions saved to {output_path}")
    print("Direct predictions completed successfully!")
    
    return True

if __name__ == "__main__":
    success = make_direct_predictions()
    if not success:
        print("Failed to make direct predictions. Please check the error messages above.") 