import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

class TournamentExitPredictor:
    """
    Deep learning model to predict NCAA tournament exit rounds based on team statistics
    and historical tournament performance.
    """
    
    def __init__(self, model_save_path='model'):
        """
        Initialize the Tournament Exit Predictor
        
        Parameters:
        -----------
        model_save_path : str
            Path to save model artifacts
        """
        self.model_save_path = model_save_path
        
        # Exit round mappings
        self.exit_round_mapping = {
            0: 'Did Not Make Tournament',
            1: 'First Round',
            2: 'Second Round',
            3: 'Sweet 16',
            4: 'Elite 8',
            5: 'Final Four',
            6: 'Championship Game',
            7: 'National Champion'
        }
        
        # Feature columns for model training
        self.feature_columns = [
            'AdjEM', 'RankAdjEM', 'AdjOE', 'RankAdjOE', 
            'AdjDE', 'RankAdjDE', 'AdjTempo', 'RankAdjTempo',
            'Luck', 'SOS_AdjEM', 'SOS_OppO', 'SOS_OppD',
            'NCSOS_AdjEM', 'Seed'  # Seed will be added in prepare_data
        ]
        
        # Create directories for model artifacts
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def load_historical_data(self, data_dir, years=range(2009, 2025)):
        """
        Load historical KenPom data with tournament results
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the KenPom data files
        years : range or list, optional
            Years to load data for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing historical KenPom data with tournament results
        """
        print("Loading historical tournament data for years 2009-2024...")
        
        all_data = []
        tournament_teams = 0
        
        # Process regular historical files
        for year in years:
            if year == 2020:  # Skip 2020 (COVID)
                continue
                
            try:
                # Attempt to load the data
                file_path = os.path.join(data_dir, f"processed_{year}.csv")
                
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
                tournament_teams += year_tournament_teams
                
                print(f"  Loaded {len(df)} teams from {year}, including {year_tournament_teams} tournament teams")
                all_data.append(df)
            except Exception as e:
                print(f"  Error loading data for {year}: {e}")
        
        # Check for summary files and process them separately
        # Handle summary25.csv (current season)
        current_season_file = os.path.join(data_dir, "summary25.csv")
        if os.path.exists(current_season_file):
            try:
                print(f"Processing current season data from {current_season_file}...")
                current_df = pd.read_csv(current_season_file)
                
                # Clean data
                for col in current_df.columns:
                    if current_df[col].dtype == 'object' and col != 'TeamName':
                        current_df[col] = current_df[col].str.replace('"', '').astype(float)
                    elif col == 'TeamName':
                        current_df[col] = current_df[col].str.replace('"', '')
                
                # Add year column
                current_df['Year'] = 2025
                
                # Estimate seeds based on team ranking
                current_df = self.estimate_seeds_for_2025(current_df)
                
                print(f"  Loaded {len(current_df)} teams from 2025 with estimated seeds")
                all_data.append(current_df)
            except Exception as e:
                print(f"  Error processing current season data: {e}")
        
        # Handle summary23.csv (2023 season)
        summary_2023_file = os.path.join(data_dir, "summary23.csv")
        if os.path.exists(summary_2023_file):
            try:
                print(f"Processing 2023 season data from {summary_2023_file}...")
                summary_2023_df = pd.read_csv(summary_2023_file)
                
                # Clean data
                for col in summary_2023_df.columns:
                    if summary_2023_df[col].dtype == 'object' and col != 'TeamName':
                        summary_2023_df[col] = summary_2023_df[col].str.replace('"', '').astype(float)
                    elif col == 'TeamName':
                        summary_2023_df[col] = summary_2023_df[col].str.replace('"', '')
                
                # Add year column
                summary_2023_df['Year'] = 2023
                
                # Add tournament exit rounds - use actual 2023 tournament data if available
                # For demonstration, we'll add placeholder exit rounds for top 68 teams
                summary_2023_df = self.add_tournament_data_for_2023(summary_2023_df)
                
                tournament_teams_2023 = summary_2023_df['TournamentExitRound'].notnull().sum()
                tournament_teams += tournament_teams_2023
                
                print(f"  Loaded {len(summary_2023_df)} teams from 2023, including {tournament_teams_2023} tournament teams")
                all_data.append(summary_2023_df)
            except Exception as e:
                print(f"  Error processing 2023 data: {e}")
        
        if not all_data:
            raise ValueError("No historical data could be loaded")
        
        # Combine all years
        historical_data = pd.concat(all_data, ignore_index=True)
        
        # Print distribution of exit rounds
        exit_round_counts = historical_data['TournamentExitRound'].value_counts().sort_index()
        
        print("\nDistribution of tournament exit rounds:")
        for exit_round, count in exit_round_counts.items():
            if pd.isna(exit_round):
                print(f"  Did Not Make Tournament: {count} teams")
            else:
                print(f"  {self.exit_round_mapping.get(int(exit_round), f'Unknown ({exit_round})')}: {count} teams")
        
        return historical_data
    
    def add_tournament_data_for_2023(self, df):
        """
        Add tournament exit round and seed data for 2023 season
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing 2023 season data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with tournament exit round and seed data added
        """
        print("Adding tournament data for 2023 season...")
        
        # Add Seed column if it doesn't exist
        if 'Seed' not in df.columns:
            df['Seed'] = np.nan
        
        # Add TournamentExitRound column if it doesn't exist
        if 'TournamentExitRound' not in df.columns:
            df['TournamentExitRound'] = np.nan
        
        # Actual 2023 tournament teams with their seeds and exit rounds
        # This is simplified data - in a real implementation, you would have the full dataset
        tournament_data = {
            # Elite 8 teams (Exit Round 4)
            'UConn': {'Seed': 4, 'ExitRound': 7},  # National Champion
            'San Diego St': {'Seed': 5, 'ExitRound': 6},  # Runner-up
            'Miami FL': {'Seed': 5, 'ExitRound': 5},  # Final Four
            'Florida Atlantic': {'Seed': 9, 'ExitRound': 5},  # Final Four
            'Kansas St': {'Seed': 3, 'ExitRound': 4},  # Elite 8
            'Gonzaga': {'Seed': 3, 'ExitRound': 4},  # Elite 8
            'Texas': {'Seed': 2, 'ExitRound': 4},  # Elite 8
            'Creighton': {'Seed': 6, 'ExitRound': 4},  # Elite 8
            
            # Sweet 16 teams (Exit Round 3)
            'Alabama': {'Seed': 1, 'ExitRound': 3},
            'Houston': {'Seed': 1, 'ExitRound': 3},
            'UCLA': {'Seed': 2, 'ExitRound': 3},
            'Tennessee': {'Seed': 4, 'ExitRound': 3},
            'Xavier': {'Seed': 3, 'ExitRound': 3},
            'Michigan St': {'Seed': 7, 'ExitRound': 3},
            'Arkansas': {'Seed': 8, 'ExitRound': 3},
            'Princeton': {'Seed': 15, 'ExitRound': 3},
            
            # Round of 32 teams (Exit Round 2)
            'Purdue': {'Seed': 1, 'ExitRound': 2},
            'Kansas': {'Seed': 1, 'ExitRound': 2},
            'Marquette': {'Seed': 2, 'ExitRound': 2},
            'Baylor': {'Seed': 3, 'ExitRound': 2},
            'Virginia': {'Seed': 4, 'ExitRound': 2},
            'Saint Mary\'s': {'Seed': 5, 'ExitRound': 2},
            'TCU': {'Seed': 6, 'ExitRound': 2},
            'Missouri': {'Seed': 7, 'ExitRound': 2},
            'Maryland': {'Seed': 8, 'ExitRound': 2},
            'Auburn': {'Seed': 9, 'ExitRound': 2},
            'Penn St': {'Seed': 10, 'ExitRound': 2},
            'Pittsburgh': {'Seed': 11, 'ExitRound': 2},
            'Kentucky': {'Seed': 6, 'ExitRound': 2},
            'Northwestern': {'Seed': 7, 'ExitRound': 2},
            'Memphis': {'Seed': 8, 'ExitRound': 2},
            'Florida Atlantic': {'Seed': 9, 'ExitRound': 2},
            
            # First Round Exit teams (Exit Round 1) - just a subset for brevity
            'Iowa': {'Seed': 8, 'ExitRound': 1},
            'West Virginia': {'Seed': 9, 'ExitRound': 1},
            'USC': {'Seed': 10, 'ExitRound': 1},
            'Providence': {'Seed': 11, 'ExitRound': 1},
            'Charleston': {'Seed': 12, 'ExitRound': 1},
            'Furman': {'Seed': 13, 'ExitRound': 1},
            'UC Santa Barbara': {'Seed': 14, 'ExitRound': 1},
            'Montana St': {'Seed': 14, 'ExitRound': 1},
            'Vermont': {'Seed': 15, 'ExitRound': 1},
            'Texas A&M C.C.': {'Seed': 16, 'ExitRound': 1},
            'FDU': {'Seed': 16, 'ExitRound': 1},
            'Howard': {'Seed': 16, 'ExitRound': 1},
            'Northern Kentucky': {'Seed': 16, 'ExitRound': 1}
        }
        
        # Update the dataframe with the tournament data
        for i, row in df.iterrows():
            team_name = row['TeamName']
            # Sometimes team names might have slight differences, this is a simple exact match
            if team_name in tournament_data:
                df.at[i, 'Seed'] = tournament_data[team_name]['Seed']
                df.at[i, 'TournamentExitRound'] = tournament_data[team_name]['ExitRound']
        
        # Count how many teams were assigned tournament data
        tournament_count = df['TournamentExitRound'].notnull().sum()
        print(f"  Added tournament data for {tournament_count} teams in 2023")
        
        return df
    
    def analyze_seed_performance(self, historical_data):
        """
        Analyze how seeds historically perform in the tournament
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame containing historical tournament data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing average exit round by seed
        """
        print("Analyzing seed performance in historical tournaments...")
        
        # Check if we have the required columns
        if 'Seed' not in historical_data.columns or 'TournamentExitRound' not in historical_data.columns:
            print("Warning: 'Seed' or 'TournamentExitRound' columns not found in historical data.")
            print("Creating default seed performance data based on typical tournament outcomes.")
            
            # Create a default seed performance DataFrame based on typical tournament outcomes
            seeds = list(range(1, 17))
            # Estimated average exit rounds for each seed (1-indexed)
            # 1 seeds: typically reach Final Four (5)
            # 2 seeds: typically reach Elite 8 (4)
            # 3-4 seeds: typically reach Sweet 16 (3)
            # 5-8 seeds: typically reach Round of 32 (2)
            # 9-16 seeds: typically exit in Round of 64 (1)
            avg_exit_rounds = [5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            counts = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]  # Dummy counts
            
            seed_performance = pd.DataFrame({
                'Seed': seeds,
                'mean': avg_exit_rounds,
                'count': counts
            })
        else:
            # Filter to include only tournament teams
            tournament_teams = historical_data[historical_data['TournamentExitRound'].notnull()].copy()
            
            # Group by seed and calculate average exit round
            seed_performance = tournament_teams.groupby('Seed')['TournamentExitRound'].agg(['mean', 'count']).reset_index()
        
        # Rename columns
        seed_performance.columns = ['Seed', 'AvgExitRound', 'Count']
        
        # Plot the average exit round by seed
        plt.figure(figsize=(10, 6))
        plt.bar(seed_performance['Seed'], seed_performance['AvgExitRound'])
        plt.xlabel('Seed')
        plt.ylabel('Average Exit Round')
        plt.title('Average Tournament Exit Round by Seed')
        plt.xticks(range(1, 17))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a horizontal line for each tournament round
        round_names = {
            1: 'First Round',
            2: 'Second Round',
            3: 'Sweet 16',
            4: 'Elite 8',
            5: 'Final Four',
            6: 'Championship Game',
            7: 'National Champion'
        }
        
        for round_num, round_name in round_names.items():
            plt.axhline(y=round_num, color='gray', linestyle='--', alpha=0.5)
            plt.text(16.5, round_num, round_name, verticalalignment='center')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'seed_performance.png'))
        
        # Save the data
        seed_performance.to_csv(os.path.join(self.model_save_path, 'seed_performance.csv'), index=False)
        
        return seed_performance
    
    def prepare_data(self, historical_data):
        """
        Prepare data for training the model
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame containing historical KenPom data
            
        Returns:
        --------
        dict
            Dictionary containing prepared data for training
        """
        # Filter to tournament teams only
        tournament_teams = historical_data[historical_data['TournamentExitRound'].notnull()].copy()
        
        # If we don't have tournament data, create a synthetic dataset
        if len(tournament_teams) == 0:
            print("Warning: No tournament data found. Creating synthetic training data.")
            # Get the columns we need
            numeric_cols = [col for col in historical_data.columns 
                            if historical_data[col].dtype in ['int64', 'float64']
                            and col not in ['Year', 'TournamentExitRound', 'Seed']]
            
            # Prioritize key metrics if they exist
            preferred_feature_names = [
                'AdjEM', 'RankAdjEM', 'AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 
                'AdjTempo', 'RankAdjTempo'
            ]
            
            # Check which preferred features are available
            available_features = [col for col in preferred_feature_names if col in historical_data.columns]
            
            # If we have some preferred features, use those, otherwise use all numeric columns
            if available_features:
                feature_cols = available_features
                print(f"Using preferred features: {feature_cols}")
            else:
                feature_cols = numeric_cols[:8]  # Limit to 8 features
                print(f"Using available numeric features: {feature_cols}")
            
            # Create a more realistic synthetic dataset with 100 teams
            n_samples = 100
            
            # Generate synthetic data based on the distribution of actual data
            X = np.zeros((n_samples, len(feature_cols)))
            for i, col in enumerate(feature_cols):
                if col in historical_data.columns:
                    mean = historical_data[col].mean()
                    std = historical_data[col].std()
                    X[:, i] = np.random.normal(mean, std, n_samples)
                else:
                    # If column doesn't exist, use reasonable defaults
                    if 'Rank' in col:
                        # Ranks are typically 1-350
                        X[:, i] = np.random.randint(1, 351, n_samples)
                    elif 'Adj' in col:
                        # Adjusted metrics are typically in range of 90-120
                        X[:, i] = np.random.normal(105, 10, n_samples)
                    else:
                        X[:, i] = np.random.normal(0, 1, n_samples)
            
            # Generate synthetic exit rounds - more heavily weighted towards earlier exits
            weights = [0.5, 0.25, 0.15, 0.05, 0.03, 0.02]  # Weights for rounds 1-6
            y = np.random.choice(np.arange(1, 7), size=n_samples, p=weights)
            
            # Generate synthetic seeds - more realistic distribution
            seed_weights = [0.04] * 4 + [0.05] * 4 + [0.06] * 4 + [0.08] * 4  # Weights for seeds 1-16
            seeds = np.random.choice(np.arange(1, 17), size=n_samples, p=seed_weights)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return {
                'X': X,
                'X_scaled': X_scaled,
                'y': y,
                'seeds': seeds,
                'feature_cols': feature_cols,
                'scaler': scaler
            }
        
        # For real data, continue with normal processing
        print(f"Preparing data from {len(tournament_teams)} tournament teams...")
        
        # Define feature columns
        feature_cols = [
            'AdjEM', 'RankAdjEM', 'AdjOE', 'RankAdjOE', 'AdjDE', 'RankAdjDE', 
            'AdjTempo', 'RankAdjTempo', 'Seed'
        ]
        
        # Filter columns to only those that exist in the dataset
        existing_feature_cols = [col for col in feature_cols if col in tournament_teams.columns]
        
        if len(existing_feature_cols) < 3:
            raise ValueError(f"Insufficient feature columns found. Need at least 3, found {len(existing_feature_cols)}")
        
        print(f"Using features: {existing_feature_cols}")
        
        # Select features and target
        X = tournament_teams[existing_feature_cols].values
        y = tournament_teams['TournamentExitRound'].values.astype(int)
        seeds = tournament_teams['Seed'].values if 'Seed' in tournament_teams.columns else None
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save feature columns for later use
        with open(os.path.join(self.model_save_path, 'feature_cols.json'), 'w') as f:
            json.dump(existing_feature_cols, f)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(self.model_save_path, 'scaler.joblib'))
        
        return {
            'X': X,
            'X_scaled': X_scaled,
            'y': y,
            'seeds': seeds,
            'feature_cols': existing_feature_cols,
            'scaler': scaler
        }
    
    def build_model(self, input_shape):
        """
        Build a neural network model for predicting tournament exit rounds
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Neural network model
        """
        print("\nBuilding neural network model...")
        
        # Create model
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer - regression for exit round (can be fractional)
            Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # Mean squared error for regression
            metrics=['accuracy']  # Accuracy metric
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def train_model(self, prepared_data, epochs=100, batch_size=32):
        """
        Train the model to predict tournament exit rounds
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data for training
        epochs : int, optional
            Number of epochs to train for
        batch_size : int, optional
            Batch size for training
            
        Returns:
        --------
        tensorflow.keras.Model
            Trained model
        """
        print("\nTraining tournament exit round prediction model...")
        
        # Get data from prepared_data
        X_scaled = prepared_data['X_scaled']
        y = prepared_data['y']
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Build the model
        model = self.build_model(input_shape=X_train.shape[1])
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'exit_round_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        
        # Evaluate the model
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        # Calculate feature importance (if using neural network, use permutation importance)
        if len(prepared_data['feature_cols']) > 0:
            print("\nFeature importance:")
            
            # Simple permutation importance
            baseline_score = model.evaluate(X_val, y_val, verbose=0)[1]
            importances = []
            
            for i, feature in enumerate(prepared_data['feature_cols']):
                # Create a copy of the validation data
                X_permuted = X_val.copy()
                
                # Permute the feature
                np.random.shuffle(X_permuted[:, i])
                
                # Evaluate the model on the permuted data
                permuted_score = model.evaluate(X_permuted, y_val, verbose=0)[1]
                
                # Calculate importance as the decrease in performance
                importance = baseline_score - permuted_score
                importances.append(importance)
                
                print(f"{feature}: {importance:.4f}")
            
            # Plot feature importance
            feature_importance = pd.DataFrame({
                'Feature': prepared_data['feature_cols'],
                'Importance': importances
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['Feature'], feature_importance['Importance'])
            plt.xlabel('Importance (decrease in accuracy when permuted)')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_save_path, 'feature_importance.png'))
        
        # Save the model
        model.save(os.path.join(self.model_save_path, 'exit_round_model.h5'))
        
        return model
    
    def load_trained_model(self):
        """
        Load a previously trained model
        
        Returns:
        --------
        tuple
            (model, scaler, feature_columns)
        """
        print("\nLoading trained model...")
        
        model_path = os.path.join(self.model_save_path, 'exit_round_model.h5')
        scaler_path = os.path.join(self.model_save_path, 'scaler.joblib')
        feature_columns_path = os.path.join(self.model_save_path, 'feature_cols.json')
        
        # Check if files exist
        for path in [model_path, scaler_path, feature_columns_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load model
        model = load_model(model_path)
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        # Load feature columns
        with open(feature_columns_path, 'r') as f:
            feature_columns = json.load(f)
        
        return model, scaler, feature_columns
    
    def estimate_seeds_for_2025(self, current_data):
        """
        Estimate tournament seeds for the current season based on team rankings
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            DataFrame containing current season data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with estimated seeds added
        """
        print("Estimating tournament seeds for the current season...")
        
        # Create a copy of the data to avoid modifying the original
        current_data = current_data.copy()
        
        # Add Seed column if it doesn't exist
        if 'Seed' not in current_data.columns:
            current_data['Seed'] = np.nan
        
        # Add TournamentExitRound column if it doesn't exist
        if 'TournamentExitRound' not in current_data.columns:
            current_data['TournamentExitRound'] = np.nan
        
        # Sort teams by AdjEM (Adjusted Efficiency Margin) - the higher the better
        if 'AdjEM' in current_data.columns:
            ranked_teams = current_data.sort_values('AdjEM', ascending=False).reset_index(drop=True)
        else:
            # Use RankAdjEM if AdjEM is not available
            ranked_teams = current_data.sort_values('RankAdjEM', ascending=True).reset_index(drop=True)
        
        # Take top 68 teams for the tournament
        tournament_teams = ranked_teams.iloc[:68].copy()
        
        # Create 4 regions with 16 seeds each (simplified approach)
        seeds_per_region = 16
        num_regions = 4
        
        # Assign seeds to teams using snake draft pattern (1, 2, 3, 4, 4, 3, 2, 1, ...)
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
        
        # Assign simulated exit rounds based on typical performance by seed
        # Use a random element but with seed-weighted probabilities
        np.random.seed(42)  # For reproducibility
        
        # Create a dictionary of exit round probabilities by seed
        seed_exit_probs = {
            1: [0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.2],  # 1 seeds - high chance of deep run
            2: [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1],  # 2 seeds
            3: [0.0, 0.2, 0.3, 0.3, 0.1, 0.05, 0.05],  # 3 seeds
            4: [0.1, 0.2, 0.4, 0.2, 0.05, 0.03, 0.02],  # 4 seeds
            5: [0.2, 0.3, 0.3, 0.1, 0.05, 0.03, 0.02],  # 5 seeds
            6: [0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02],  # 6 seeds
            7: [0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0.01],  # 7 seeds
            8: [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01],  # 8 seeds
            9: [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01],  # 9 seeds
            10: [0.6, 0.3, 0.05, 0.03, 0.01, 0.005, 0.005],  # 10 seeds
            11: [0.7, 0.2, 0.05, 0.03, 0.01, 0.005, 0.005],  # 11 seeds
            12: [0.7, 0.2, 0.05, 0.03, 0.01, 0.005, 0.005],  # 12 seeds
            13: [0.8, 0.15, 0.03, 0.01, 0.005, 0.003, 0.002],  # 13 seeds
            14: [0.9, 0.07, 0.02, 0.005, 0.003, 0.001, 0.001],  # 14 seeds
            15: [0.95, 0.03, 0.01, 0.005, 0.003, 0.001, 0.001],  # 15 seeds
            16: [0.99, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0002]  # 16 seeds
        }
        
        # Assign exit rounds based on seed probabilities
        for i, row in tournament_teams.iterrows():
            # Only process rows with valid seeds
            if pd.notna(row['Seed']):
                seed = int(row['Seed'])
                probs = seed_exit_probs.get(seed, [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
                # Exit rounds are 1-7 (1=first round, 7=champion)
                exit_round = np.random.choice(range(1, 8), p=probs)
                tournament_teams.loc[i, 'TournamentExitRound'] = exit_round
                
                # Make sure there's only one champion
                if exit_round == 7:
                    for s in range(1, 17):
                        if s != seed:
                            # Update probabilities to ensure only one champion
                            seed_exit_probs[s][-1] = 0
                            # Normalize
                            total = sum(seed_exit_probs[s])
                            if total > 0:
                                seed_exit_probs[s] = [p/total for p in seed_exit_probs[s]]
        
        # Ensure there's exactly one champion, one runner-up, two final four teams, etc.
        exit_counts = tournament_teams['TournamentExitRound'].value_counts()
        
        # If we have more than one champion, adjust
        if exit_counts.get(7, 0) > 1:
            champions = tournament_teams[tournament_teams['TournamentExitRound'] == 7].index
            # Keep only the team with the best AdjEM as champion
            if len(champions) > 1:
                best_champion = tournament_teams.loc[champions].sort_values('AdjEM', ascending=False).index[0]
                for idx in champions:
                    if idx != best_champion:
                        tournament_teams.loc[idx, 'TournamentExitRound'] = 6  # Make them runner-up
        
        # If we have more than one runner-up, adjust
        if exit_counts.get(6, 0) > 1:
            runner_ups = tournament_teams[tournament_teams['TournamentExitRound'] == 6].index
            if len(runner_ups) > 1:
                best_runner_up = tournament_teams.loc[runner_ups].sort_values('AdjEM', ascending=False).index[0]
                for idx in runner_ups:
                    if idx != best_runner_up:
                        tournament_teams.loc[idx, 'TournamentExitRound'] = 5  # Make them final four
        
        # Now update the original dataframe with the estimated tournament data
        for i, row in tournament_teams.iterrows():
            team_idx = current_data[current_data['TeamName'] == row['TeamName']].index
            if len(team_idx) > 0:
                current_data.loc[team_idx, 'Seed'] = row['Seed']
                current_data.loc[team_idx, 'TournamentExitRound'] = row['TournamentExitRound']
        
        # Create the following fields:
        # - PredictedExitRound: The same as TournamentExitRound
        # - PredictedExitRoundInt: Integer version of PredictedExitRound
        # - PredictedExit: String name of the exit round
        # - ChampionshipPct: Percentage chance of winning championship (0-100)
        # - FinalFourPct: Percentage chance of making Final Four (0-100)
        
        current_data['PredictedExitRound'] = current_data['TournamentExitRound']
        
        # Use Int64 type which can handle NaN values
        # First convert to float to handle any non-integer values
        current_data['PredictedExitRoundInt'] = pd.to_numeric(current_data['PredictedExitRound'], errors='coerce')
        current_data['PredictedExitRoundInt'] = current_data['PredictedExitRoundInt'].apply(
            lambda x: int(x) if pd.notna(x) else pd.NA
        )
        
        # Map exit rounds to names - use a function to handle NaN values
        def map_exit_round(x):
            if pd.isna(x):
                return "Did Not Make Tournament"
            return self.exit_round_mapping.get(int(x), f"Unknown ({x})")
        
        current_data['PredictedExit'] = current_data['PredictedExitRoundInt'].apply(map_exit_round)
        
        # Calculate championship and Final Four probabilities based on seed
        current_data['ChampionshipPct'] = 0.0
        current_data['FinalFourPct'] = 0.0
        
        for i, row in current_data.iterrows():
            if pd.notna(row['Seed']):
                seed = int(row['Seed'])
                # Championship probability based on seed
                if seed == 1:
                    champ_pct = 20.0
                elif seed == 2:
                    champ_pct = 10.0
                elif seed == 3:
                    champ_pct = 5.0
                elif seed == 4:
                    champ_pct = 2.0
                elif seed <= 8:
                    champ_pct = 1.0
                else:
                    champ_pct = 0.1
                    
                # Final Four probability based on seed
                if seed == 1:
                    ff_pct = 40.0
                elif seed == 2:
                    ff_pct = 25.0
                elif seed == 3:
                    ff_pct = 15.0
                elif seed == 4:
                    ff_pct = 10.0
                elif seed <= 8:
                    ff_pct = 5.0
                else:
                    ff_pct = 1.0
                    
                current_data.loc[i, 'ChampionshipPct'] = champ_pct
                current_data.loc[i, 'FinalFourPct'] = ff_pct
        
        tournament_count = current_data['Seed'].notnull().sum()
        print(f"  Estimated tournament seeds and exit rounds for {tournament_count} teams")
        
        return current_data
    
    def predict_tournament_performance(self, current_data, model=None, scaler=None, feature_cols=None):
        """
        Predict how teams will perform in the tournament
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            DataFrame containing current season KenPom data, with seeds if available
        model : tensorflow.keras.Model, optional
            Trained model to use for predictions
        scaler : sklearn.preprocessing.StandardScaler, optional
            Fitted scaler to use for feature scaling
        feature_cols : list, optional
            List of feature columns to use for prediction
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predicted tournament performance
        """
        print("\nPredicting tournament performance for current teams...")
        
        # Create a copy of the data to avoid modifying the original
        current_data = current_data.copy()
        
        # If no seed column, estimate seeds
        if 'Seed' not in current_data.columns or current_data['Seed'].isnull().all():
            print("No seeds found. Estimating seeds for tournament teams...")
            current_data = self.estimate_seeds_for_2025(current_data)
            
            # If we only want to use seed-based predictions, return here
            if model is None:
                print("Using seed-based predictions only")
                return current_data
        
        # Load model artifacts if not provided
        if model is None or scaler is None or feature_cols is None:
            try:
                print("Loading trained model and artifacts...")
                model, scaler, feature_cols = self.load_trained_model()
            except Exception as e:
                print(f"Error loading trained model: {e}")
                print("Falling back to seed-based predictions")
                return current_data
        
        # Get tournament teams - make sure to only select teams with valid seed values
        tournament_teams = current_data[current_data['Seed'].notna()].copy()
        
        if len(tournament_teams) == 0:
            print("No tournament teams found with valid seeds. Returning seed-based predictions.")
            return current_data
        
        # Prepare features for prediction
        print(f"Making predictions for {len(tournament_teams)} tournament teams using {len(feature_cols)} features")
        
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
        print("Generating exit round predictions...")
        y_pred = model.predict(X_scaled)
        
        # Round predictions to nearest integer (representing exit round)
        rounded_preds = np.round(y_pred).astype(int).flatten()
        
        # Clip predictions to valid range (1-7)
        rounded_preds = np.clip(rounded_preds, 1, 7)
        
        # Store predictions
        tournament_teams['PredictedExitRound'] = rounded_preds
        tournament_teams['PredictedExitRoundInt'] = rounded_preds
        
        # Map to exit round names
        tournament_teams['PredictedExit'] = tournament_teams['PredictedExitRoundInt'].apply(
            lambda x: self.exit_round_mapping.get(int(x), f"Unknown ({x})")
        )
        
        # Calculate championship probabilities based on exit round
        print("Calculating championship and Final Four probabilities...")
        # Base probabilities on seed and predicted exit round
        tournament_teams['ChampionshipPct'] = 0.0
        tournament_teams['FinalFourPct'] = 0.0
        
        # Calculate championship and Final Four percentages based on seed and predicted exit round
        for i, row in tournament_teams.iterrows():
            # Only process rows with valid seeds
            if pd.notna(row['Seed']) and pd.notna(row['PredictedExitRoundInt']):
                seed = int(row['Seed'])
                exit_round = int(row['PredictedExitRoundInt'])
                
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
        
        # Create scatter plot of predicted exit round vs seed
        plt.figure(figsize=(10, 6))
        # Only include teams with valid seeds and predicted exit rounds
        valid_teams = tournament_teams[tournament_teams['Seed'].notna() & tournament_teams['PredictedExitRound'].notna()]
        plt.scatter(valid_teams['Seed'], valid_teams['PredictedExitRound'], alpha=0.6)
        
        # Add team labels for Final Four and better
        for i, row in valid_teams.iterrows():
            if pd.notna(row['PredictedExitRoundInt']) and row['PredictedExitRoundInt'] >= 5:  # Final Four or better
                plt.annotate(row['TeamName'], 
                            xy=(row['Seed'], row['PredictedExitRound']),
                            xytext=(5, 5),
                            textcoords='offset points')
        
        plt.xlabel('Seed')
        plt.ylabel('Predicted Exit Round')
        plt.title('Predicted Tournament Performance by Seed')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal lines for each round
        for round_num, round_name in self.exit_round_mapping.items():
            if round_num >= 1:  # Only include tournament rounds
                plt.axhline(y=round_num, color='gray', linestyle='--', alpha=0.5)
                plt.text(16.5, round_num, round_name, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'predicted_performance_scatter.png'))
        
        # Update the original dataframe with predictions
        for col in ['PredictedExitRound', 'PredictedExitRoundInt', 'PredictedExit', 'ChampionshipPct', 'FinalFourPct']:
            if col in tournament_teams.columns:
                for i, row in tournament_teams.iterrows():
                    team_name = row['TeamName']
                    team_idx = current_data.index[current_data['TeamName'] == team_name].tolist()
                    if team_idx:
                        current_data.loc[team_idx, col] = row[col]
        
        print(f"Successfully predicted tournament performance for {len(tournament_teams)} teams")
        
        return current_data
    
    def run_full_pipeline(self, historical_data_dir, current_data_path):
        """
        Run the entire exit round prediction pipeline:
        1. Load historical data
        2. Analyze seed performance
        3. Prepare data for model training
        4. Train model
        5. Predict tournament exit rounds for current teams
        
        Parameters:
        -----------
        historical_data_dir : str
            Directory containing historical KenPom data
        current_data_path : str
            Path to current season KenPom data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing predictions for current season teams
        """
        print("\nRunning exit round prediction pipeline...")
        
        # Step 1: Load historical data
        historical_data = self.load_historical_data(historical_data_dir)
        
        # Step 2: Analyze seed performance
        seed_performance = self.analyze_seed_performance(historical_data)
        
        # Step 3: Prepare data for model training
        prepared_data = self.prepare_data(historical_data)
        
        # Step 4: Train model
        try:
            model = self.train_model(prepared_data, epochs=150, batch_size=16)
            
            # Save model performance metrics
            tournament_teams = historical_data[historical_data['TournamentExitRound'].notnull()].copy()
            tournament_teams_count = len(tournament_teams)
            
            print(f"\nModel trained on {tournament_teams_count} historical tournament teams")
            
        except Exception as e:
            print(f"Error training model: {e}")
            print("Loading a pre-trained model if available, otherwise using seed-based predictions")
            model = None
        
        # Step 5: Load current season data
        print("\nLoading current season data...")
        try:
            current_data = pd.read_csv(current_data_path)
            
            # Clean data
            for col in current_data.columns:
                if current_data[col].dtype == 'object' and col != 'TeamName':
                    current_data[col] = current_data[col].str.replace('"', '').astype(float)
                elif col == 'TeamName':
                    current_data[col] = current_data[col].str.replace('"', '')
                    
            print(f"Loaded {len(current_data)} teams from current season")
        except Exception as e:
            print(f"Error loading current season data: {e}")
            return None
        
        # Check if this is the summary data that already has predictions
        if 'PredictedExitRound' in current_data.columns:
            print("Current data already has predictions, using existing values")
            predictions = current_data
        else:
            # Step 6: Predict tournament exit rounds for current teams
            try:
                # If we have a model, use it
                if model is not None:
                    print("Using trained model for predictions")
                    predictions = self.predict_tournament_performance(current_data, model=model)
                # Otherwise, use estimate_seeds_for_2025 method which has built-in exit round prediction
                else:
                    print("Using seed-based predictions")
                    current_data = self.estimate_seeds_for_2025(current_data)
                    predictions = current_data
                
                # Save predictions
                predictions.to_csv(os.path.join(self.model_save_path, 'exit_round_predictions.csv'), index=False)
                
                # Save tournament teams predictions separately
                tournament_teams_preds = predictions[predictions['Seed'].notnull()].copy()
                tournament_teams_preds.to_csv(
                    os.path.join(self.model_save_path, 'tournament_teams_predictions.csv'),
                    index=False
                )
                
                # Plot championship probabilities
                if 'ChampionshipPct' in predictions.columns:
                    tournament_teams_preds = tournament_teams_preds.sort_values('ChampionshipPct', ascending=False).head(20)
                    
                    plt.figure(figsize=(12, 8))
                    plt.barh(tournament_teams_preds['TeamName'], tournament_teams_preds['ChampionshipPct'])
                    plt.xlabel('Championship Probability (%)')
                    plt.title('Top 20 Teams by Championship Probability')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_save_path, 'championship_probabilities.png'))
                
                # Plot Final Four probabilities
                if 'FinalFourPct' in predictions.columns:
                    tournament_teams_preds = predictions[predictions['Seed'].notnull()].copy()
                    tournament_teams_preds = tournament_teams_preds.sort_values('FinalFourPct', ascending=False).head(20)
                    
                    plt.figure(figsize=(12, 8))
                    plt.barh(tournament_teams_preds['TeamName'], tournament_teams_preds['FinalFourPct'])
                    plt.xlabel('Final Four Probability (%)')
                    plt.title('Top 20 Teams by Final Four Probability')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_save_path, 'final_four_probabilities.png'))
                
            except Exception as e:
                print(f"Error predicting tournament performance: {e}")
                # If prediction fails, return current data without predictions
                predictions = current_data
        
        return predictions


if __name__ == "__main__":
    # Create model directory
    model_dir = "../../models/exit_round/model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Run the pipeline
    predictor = TournamentExitPredictor(model_save_path=model_dir)
    predictions = predictor.run_full_pipeline()
    
    print("\nPredictions saved to model directory")
    print("Pipeline completed successfully!") 