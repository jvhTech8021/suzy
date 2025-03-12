import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def train_deep_learning_model():
    """
    Train a deep learning model to predict NCAA tournament exit rounds
    using historical data and make predictions for the current season.
    """
    print("NCAA Tournament Exit Round Prediction - Deep Learning Model")
    print("=" * 80)
    
    # Paths
    model_save_path = 'deep_learning_model'
    historical_data_dir = 'susan_kenpom'
    current_data_path = 'susan_kenpom/summary25.csv'
    output_path = os.path.join(model_save_path, 'deep_learning_predictions.csv')
    
    # Create directories
    os.makedirs(model_save_path, exist_ok=True)
    
    # Step 1: Load historical data
    print("\nStep 1: Loading historical data...")
    
    all_data = []
    tournament_teams_count = 0
    
    # Process historical files (2009-2024)
    for year in range(2009, 2025):
        if year == 2020:  # Skip 2020 (COVID)
            continue
            
        try:
            # Attempt to load the data
            file_path = os.path.join(historical_data_dir, f"processed_{year}.csv")
            
            # Skip files that don't exist or backup files
            if not os.path.exists(file_path) or '.bak' in file_path:
                print(f"  Warning: No processed data file found for {year}")
                continue
            
            df = pd.read_csv(file_path)
            
            # Clean data - remove quotes from numeric columns
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'TeamName':
                    if isinstance(df[col].iloc[0], str):
                        df[col] = df[col].str.replace('"', '').astype(float)
                elif col == 'TeamName':
                    if isinstance(df[col].iloc[0], str):
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
    
    print(f"\nTotal historical data: {len(historical_data)} teams")
    print(f"Total tournament teams: {tournament_teams_count}")
    
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
    
    # Print distribution of tournament exit rounds
    exit_round_counts = historical_data['TournamentExitRound'].value_counts().sort_index()
    print("\nDistribution of tournament exit rounds:")
    for exit_round, count in exit_round_counts.items():
        exit_name = exit_round_mapping.get(int(exit_round), f"Unknown ({exit_round})")
        print(f"  {exit_name}: {count} teams")
    
    # Step 2: Prepare data for training
    print("\nStep 2: Preparing data for training...")
    
    # Filter to tournament teams only
    tournament_teams = historical_data[historical_data['TournamentExitRound'].notnull()].copy()
    
    # Define feature columns
    feature_cols = [
        'AdjEM', 'RankAdjEM', 'AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 
        'AdjTempo', 'RankAdjTempo', ' "seed"'
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
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Step 3: Build and train the deep learning model
    print("\nStep 3: Building and training the deep learning model...")
    
    # Define model architecture
    input_dim = X_train.shape[1]
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer - regression for exit round (can be fractional)
        Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error
    )
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate the model
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nTraining loss (MSE): {train_loss:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Validation loss (MSE): {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_history.png'))
    
    # Save the model
    model_path = os.path.join(model_save_path, 'exit_round_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Step 4: Load current season data
    print("\nStep 4: Loading current season data...")
    current_data = pd.read_csv(current_data_path)
    
    # Clean data
    for col in current_data.columns:
        if current_data[col].dtype == 'object' and col != 'TeamName':
            if isinstance(current_data[col].iloc[0], str):
                current_data[col] = current_data[col].str.replace('"', '').astype(float)
        elif col == 'TeamName':
            if isinstance(current_data[col].iloc[0], str):
                current_data[col] = current_data[col].str.replace('"', '')
    
    # Step 5: Estimate seeds for current teams
    print("\nStep 5: Estimating seeds for current teams...")
    
    # Sort by adjusted efficiency margin
    if 'AdjEM' in current_data.columns:
        ranked_teams = current_data.sort_values('AdjEM', ascending=False).reset_index(drop=True)
    else:
        ranked_teams = current_data.sort_values('RankAdjEM', ascending=True).reset_index(drop=True)
    
    # Add Seed column if it doesn't exist
    if ' "seed"' not in current_data.columns:
        current_data[' "seed"'] = np.nan
    
    # Take top 68 teams (68-team tournament format)
    tournament_teams = ranked_teams.iloc[:68].copy()
    
    # Assign seeds (1-16 for 4 regions, plus First Four)
    seeds_per_region = 16
    num_regions = 4
    
    # First, assign seeds 1-16 to the top 64 teams
    for i in range(seeds_per_region):
        seed_value = i + 1
        for j in range(num_regions):
            idx = i * num_regions + j
            if idx < 64:  # Only assign to the top 64
                tournament_teams.iloc[idx, tournament_teams.columns.get_loc(' "seed"')] = seed_value
    
    # Then, assign seed 11 or 16 to the remaining 4 teams (First Four)
    for i in range(64, 68):
        if i < 66:  # Two teams get seed 16
            tournament_teams.iloc[i, tournament_teams.columns.get_loc(' "seed"')] = 16
        else:  # Two teams get seed 11
            tournament_teams.iloc[i, tournament_teams.columns.get_loc(' "seed"')] = 11
    
    # Update the original dataframe with the estimated tournament data
    for i, row in tournament_teams.iterrows():
        team_idx = current_data[current_data['TeamName'] == row['TeamName']].index
        if len(team_idx) > 0:
            current_data.loc[team_idx, ' "seed"'] = row[' "seed"']
    
    # Step 6: Make predictions
    print("\nStep 6: Making predictions for current teams...")
    
    # Get tournament teams for prediction
    tournament_teams = current_data[current_data[' "seed"'].notna()].copy()
    
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
    rounded_preds = np.round(y_pred).astype(int).flatten()
    
    # Clip predictions to valid range (1-7)
    rounded_preds = np.clip(rounded_preds, 1, 7)
    
    # Store predictions
    tournament_teams['PredictedExitRound'] = rounded_preds
    
    # Map exit rounds to names
    tournament_teams['PredictedExit'] = tournament_teams['PredictedExitRound'].apply(
        lambda x: exit_round_mapping.get(int(x), f"Unknown ({x})")
    )
    
    # Calculate championship probabilities based on exit round and seed
    print("Calculating championship and Final Four probabilities...")
    tournament_teams['ChampionshipPct'] = 0.0
    tournament_teams['FinalFourPct'] = 0.0
    
    for i, row in tournament_teams.iterrows():
        if pd.notna(row[' "seed"']) and pd.notna(row['PredictedExitRound']):
            seed = int(row[' "seed"'])
            exit_round = int(row['PredictedExitRound'])
            
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
                
            tournament_teams.loc[i, 'ChampionshipPct'] = champ_pct
            tournament_teams.loc[i, 'FinalFourPct'] = ff_pct
    
    # Save tournament teams predictions
    tournament_teams.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Deep Learning Model Prediction Summary")
    print("=" * 80)
    print(f"Total teams predicted to make tournament: {len(tournament_teams)}")
    
    # Print distribution of predicted exit rounds
    exit_round_counts = tournament_teams['PredictedExitRound'].value_counts().sort_index()
    print("\nPredicted Exit Round Distribution:")
    for exit_round, count in exit_round_counts.items():
        exit_name = exit_round_mapping.get(int(exit_round), f"Unknown ({exit_round})")
        print(f"  {exit_name}: {count} teams")
    
    # Print top 10 championship contenders
    print("\nTop 10 Championship Contenders:")
    top_champions = tournament_teams.sort_values('ChampionshipPct', ascending=False).head(10)
    for i, (_, row) in enumerate(top_champions.iterrows(), 1):
        print(f"{i}. {row['TeamName']} (Seed {int(row[' "seed"'])}): {row['ChampionshipPct']}% championship, Predicted: {row['PredictedExit']}")
    
    # Print top 10 Final Four contenders
    print("\nTop 10 Final Four Contenders:")
    top_ff = tournament_teams.sort_values('FinalFourPct', ascending=False).head(10)
    for i, (_, row) in enumerate(top_ff.iterrows(), 1):
        print(f"{i}. {row['TeamName']} (Seed {int(row[' "seed"'])}): {row['FinalFourPct']}% Final Four")
    
    print(f"\nDeep learning model predictions saved to {output_path}")
    print("Deep learning predictions completed successfully!")
    
    return True

if __name__ == "__main__":
    success = train_deep_learning_model()
    if not success:
        print("Failed to train deep learning model. Please check the error messages above.") 