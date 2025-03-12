import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_sklearn_predictions():
    """
    Train a scikit-learn model and make predictions for tournament teams
    """
    # Paths
    model_save_path = 'march_madness_predictor/models/exit_round/model'
    scaler_path = os.path.join(model_save_path, 'sklearn_scaler.joblib')
    model_path = os.path.join(model_save_path, 'sklearn_model.joblib')
    feature_cols_path = os.path.join(model_save_path, 'sklearn_feature_cols.json')
    historical_data_dir = 'susan_kenpom'
    current_data_path = 'susan_kenpom/summary25.csv'
    output_path = os.path.join(model_save_path, 'sklearn_predictions.csv')
    
    # Create directories
    os.makedirs(model_save_path, exist_ok=True)
    
    print("Step 1: Loading historical data...")
    
    # Load historical data
    all_data = []
    tournament_teams_count = 0
    
    # Process regular historical files
    for year in range(2009, 2025):
        if year == 2020:  # Skip 2020 (COVID)
            continue
            
        try:
            # Attempt to load the data
            file_path = os.path.join(historical_data_dir, f"processed_{year}.csv")
            
            # Skip files that don't exist
            if not os.path.exists(file_path):
                print(f"  Warning: No processed data file found for {year}")
                continue
            
            df = pd.read_csv(file_path)
            
            # Clean data - remove quotes from numeric columns
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'TeamName':
                    df[col] = df[col].str.replace('"', '').astype(float)
                elif col == 'TeamName':
                    df[col] = df[col].str.replace('"', '')
            
            # Add year column
            df['Year'] = year
            
            # Count tournament teams (those with exit rounds)
            year_tournament_teams = df['TournamentExitRound'].notnull().sum() if 'TournamentExitRound' in df.columns else 0
            tournament_teams_count += year_tournament_teams
            
            print(f"  Loaded {len(df)} teams from {year}, including {year_tournament_teams} tournament teams")
            all_data.append(df)
        except Exception as e:
            print(f"  Error loading data for {year}: {e}")
    
    # Combine all years
    historical_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\nTotal tournament teams: {tournament_teams_count}")
    
    # Exit round mappings
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
    
    # Step 2: Prepare data for training
    print("\nStep 2: Preparing data for training...")
    
    # Filter to tournament teams only
    tournament_teams = historical_data[historical_data['TournamentExitRound'].notnull()].copy()
    
    # Define feature columns
    feature_cols = [
        'AdjEM', 'RankAdjEM', 'AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 
        'AdjTempo', 'RankAdjTempo', 'Seed'
    ]
    
    # Filter columns to only those that exist in the dataset
    existing_feature_cols = [col for col in feature_cols if col in tournament_teams.columns]
    
    print(f"Using features: {existing_feature_cols}")
    
    # Select features and target
    X = tournament_teams[existing_feature_cols].values
    y = tournament_teams['TournamentExitRound'].values.astype(int)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    
    # Save feature columns for later use
    with open(feature_cols_path, 'w') as f:
        json.dump(existing_feature_cols, f)
    
    # Step 3: Train the model using Random Forest Regressor
    print("\nStep 3: Training the model using Random Forest Regressor...")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"Training R^2 score: {train_score:.4f}")
    print(f"Validation R^2 score: {val_score:.4f}")
    
    # Calculate feature importance
    feature_importances = model.feature_importances_
    print("\nFeature importance:")
    for feature, importance in zip(existing_feature_cols, feature_importances):
        print(f"{feature}: {importance:.4f}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Step 4: Load current season data and make predictions
    print("\nStep 4: Loading current season data...")
    current_data = pd.read_csv(current_data_path)
    
    # Clean data
    for col in current_data.columns:
        if current_data[col].dtype == 'object' and col != 'TeamName':
            current_data[col] = current_data[col].str.replace('"', '').astype(float)
        elif col == 'TeamName':
            current_data[col] = current_data[col].str.replace('"', '')
    
    # Step 5: Estimate seeds for current teams
    print("\nStep 5: Estimating seeds for current teams...")
    
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
    
    # Add Seed column if it doesn't exist in tournament_teams
    if 'Seed' not in tournament_teams.columns:
        tournament_teams['Seed'] = np.nan
    
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
    
    # Step 6: Make predictions
    print("\nStep 6: Making predictions for current teams...")
    
    # Get tournament teams for prediction
    tournament_teams = current_data[current_data['Seed'].notna()].copy()
    
    # Ensure all feature columns exist
    missing_cols = [col for col in existing_feature_cols if col not in tournament_teams.columns]
    if missing_cols:
        print(f"Warning: Missing feature columns: {missing_cols}")
        print("Adding missing columns with default values (0)")
        for col in missing_cols:
            tournament_teams[col] = 0
    
    # Get features for prediction
    X_pred = tournament_teams[existing_feature_cols].values
    
    # Handle any missing values
    X_pred = np.nan_to_num(X_pred, nan=0.0)
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    y_pred = model.predict(X_pred_scaled)
    
    # Round predictions to nearest integer (representing exit round)
    rounded_preds = np.round(y_pred).astype(int)
    
    # Clip predictions to valid range (1-7)
    rounded_preds = np.clip(rounded_preds, 1, 7)
    
    # Store predictions
    tournament_teams['SklearnPredictedExitRound'] = rounded_preds
    
    # Map exit rounds to names
    tournament_teams['SklearnPredictedExit'] = tournament_teams['SklearnPredictedExitRound'].apply(
        lambda x: exit_round_mapping.get(int(x), f"Unknown ({x})")
    )
    
    # Calculate championship probabilities based on exit round
    print("Calculating championship and Final Four probabilities...")
    tournament_teams['SklearnChampionshipPct'] = 0.0
    tournament_teams['SklearnFinalFourPct'] = 0.0
    
    for i, row in tournament_teams.iterrows():
        if pd.notna(row['Seed']) and pd.notna(row['SklearnPredictedExitRound']):
            seed = int(row['Seed'])
            exit_round = int(row['SklearnPredictedExitRound'])
            
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
                
            tournament_teams.loc[i, 'SklearnChampionshipPct'] = champ_pct
            tournament_teams.loc[i, 'SklearnFinalFourPct'] = ff_pct
    
    # Save tournament teams predictions
    tournament_teams.to_csv(output_path, index=False)
    
    # Print summary
    print("\n================================================================================")
    print("Random Forest Model Prediction Summary")
    print("================================================================================")
    print(f"Total teams predicted to make tournament: {len(tournament_teams)}")
    
    # Print distribution of predicted exit rounds
    exit_round_counts = tournament_teams['SklearnPredictedExitRound'].value_counts().sort_index()
    print("\nPredicted Exit Round Distribution:")
    for exit_round, count in exit_round_counts.items():
        exit_name = exit_round_mapping.get(int(exit_round), f"Unknown ({exit_round})")
        print(f"  {exit_name}: {count} teams")
    
    # Print top 10 championship contenders
    print("\nTop 10 Championship Contenders:")
    top_champions = tournament_teams.sort_values('SklearnChampionshipPct', ascending=False).head(10)
    for i, (_, row) in enumerate(top_champions.iterrows(), 1):
        print(f"{i}. {row['TeamName']} (Seed {int(row['Seed'])}): {row['SklearnChampionshipPct']}% championship, Predicted: {row['SklearnPredictedExit']}")
    
    # Print top 10 Final Four contenders
    print("\nTop 10 Final Four Contenders:")
    top_ff = tournament_teams.sort_values('SklearnFinalFourPct', ascending=False).head(10)
    for i, (_, row) in enumerate(top_ff.iterrows(), 1):
        print(f"{i}. {row['TeamName']} (Seed {int(row['Seed'])}): {row['SklearnFinalFourPct']}% Final Four")
    
    print(f"\nScikit-learn model predictions saved to {output_path}")
    print("Random Forest predictions completed successfully!")
    
    return True

if __name__ == "__main__":
    success = run_sklearn_predictions()
    if not success:
        print("Failed to run scikit-learn predictions. Please check the error messages above.") 