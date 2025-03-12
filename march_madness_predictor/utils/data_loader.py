import os
import pandas as pd
import numpy as np
import json

class DataLoader:
    """
    Utility class for loading and processing data for the dashboard
    """
    
    def __init__(self, base_path=None):
        """
        Initialize the DataLoader with paths to data files
        
        Parameters:
        -----------
        base_path : str, optional
            Base path to the project directory
        """
        if base_path is None:
            # Default to the workspace root (3 levels up from this file)
            self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_path = base_path
        
        # Set paths
        self.kenpom_data_dir = os.path.join(self.base_path, "susan_kenpom")
        self.champion_profile_dir = os.path.join(self.base_path, "march_madness_predictor/models/champion_profile/model")
        self.exit_round_dir = os.path.join(self.base_path, "march_madness_predictor/models/exit_round/model")
        
        # Cache for loaded data
        self._cache = {}
    
    def get_current_season_data(self):
        """
        Load the current season KenPom data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing current season KenPom data
        """
        if "current_season" in self._cache:
            return self._cache["current_season"]
        
        file_path = os.path.join(self.kenpom_data_dir, "summary25.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Current season data not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Clean data
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'TeamName':
                df[col] = df[col].str.replace('"', '').astype(float)
            elif col == 'TeamName':
                df[col] = df[col].str.replace('"', '')
        
        self._cache["current_season"] = df
        return df
    
    def get_historical_data(self, year):
        """
        Load historical KenPom data for a specific year
        
        Parameters:
        -----------
        year : int
            Year to load data for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing historical KenPom data
        """
        cache_key = f"historical_{year}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = os.path.join(self.kenpom_data_dir, f"processed_{year}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Historical data for {year} not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Clean data
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'TeamName':
                df[col] = df[col].str.replace('"', '').astype(float)
            elif col == 'TeamName':
                df[col] = df[col].str.replace('"', '')
        
        self._cache[cache_key] = df
        return df
    
    def get_champion_profile(self):
        """
        Load the champion profile data
        
        Returns:
        --------
        dict
            Dictionary containing champion profile metrics
        """
        if "champion_profile" in self._cache:
            return self._cache["champion_profile"]
        
        file_path = os.path.join(self.champion_profile_dir, "champion_profile.json")
        
        if not os.path.exists(file_path):
            print(f"WARNING: Champion profile data not found at {file_path}")
            # Return a default champion profile if file not found
            default_profile = {
                "AdjEM": 28.72,
                "RankAdjEM": 5.4,
                "AdjOE": 120.6,
                "AdjDE": 91.8
            }
            self._cache["champion_profile"] = default_profile
            return default_profile
        
        try:
            with open(file_path, 'r') as f:
                file_content = f.read().strip()
                if not file_content:
                    raise ValueError("Empty JSON file")
                
                # Remove any trailing characters that might cause JSON parsing issues
                if file_content.endswith('%'):
                    file_content = file_content[:-1]
                
                champion_profile = json.loads(file_content)
                print(f"Successfully loaded champion profile: {champion_profile}")
            
            self._cache["champion_profile"] = champion_profile
            return champion_profile
        except Exception as e:
            print(f"ERROR loading champion profile: {str(e)}")
            # Return a default champion profile if there's an error
            default_profile = {
                "AdjEM": 28.72,
                "RankAdjEM": 5.4,
                "AdjOE": 120.6,
                "AdjDE": 91.8
            }
            self._cache["champion_profile"] = default_profile
            return default_profile
    
    def get_champion_profile_predictions(self):
        """
        Load the champion profile predictions
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing champion profile predictions
        """
        if "champion_profile_predictions" in self._cache:
            return self._cache["champion_profile_predictions"]
        
        file_path = os.path.join(self.champion_profile_dir, "all_teams_champion_profile.csv")
        
        if not os.path.exists(file_path):
            print(f"WARNING: Champion profile predictions not found at {file_path}")
            # Create a minimal DataFrame with the current season teams if file not found
            try:
                current_season = self.get_current_season_data()
                teams = current_season['TeamName'].unique()
                
                # Create a minimal DataFrame with required columns
                df = pd.DataFrame({
                    'SimilarityRank': range(1, len(teams) + 1),
                    'TeamName': teams,
                    'SimilarityPct': [80.0] * len(teams),
                    'ChampionPct': [5.0] * len(teams),
                    'FinalFourPct': [20.0] * len(teams),
                    'AdjEM': current_season['AdjEM'].values[:len(teams)],
                    'AdjOE': current_season['AdjO'].values[:len(teams)],
                    'AdjDE': current_season['AdjD'].values[:len(teams)]
                })
                
                self._cache["champion_profile_predictions"] = df
                return df
            except Exception as e:
                print(f"ERROR creating default predictions: {str(e)}")
                # Return an empty DataFrame if all else fails
                df = pd.DataFrame(columns=[
                    'SimilarityRank', 'TeamName', 'SimilarityPct', 
                    'ChampionPct', 'FinalFourPct', 'AdjEM', 'AdjOE', 'AdjDE'
                ])
                self._cache["champion_profile_predictions"] = df
                return df
        
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded champion profile predictions with {len(df)} rows")
            
            # Ensure required columns exist
            required_columns = ['TeamName', 'SimilarityPct', 'ChampionPct', 'FinalFourPct']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'TeamName':
                        # Try to find an alternative column for team names
                        team_cols = [c for c in df.columns if 'team' in c.lower()]
                        if team_cols:
                            df['TeamName'] = df[team_cols[0]]
                        else:
                            df['TeamName'] = [f"Team {i}" for i in range(len(df))]
                    else:
                        # Add default values for missing columns
                        df[col] = 0.0
            
            self._cache["champion_profile_predictions"] = df
            return df
        except Exception as e:
            print(f"ERROR loading champion profile predictions: {str(e)}")
            # Create a minimal DataFrame with the current season teams if there's an error
            try:
                current_season = self.get_current_season_data()
                teams = current_season['TeamName'].unique()
                
                # Create a minimal DataFrame with required columns
                df = pd.DataFrame({
                    'SimilarityRank': range(1, len(teams) + 1),
                    'TeamName': teams,
                    'SimilarityPct': [80.0] * len(teams),
                    'ChampionPct': [5.0] * len(teams),
                    'FinalFourPct': [20.0] * len(teams),
                    'AdjEM': current_season['AdjEM'].values[:len(teams)],
                    'AdjOE': current_season['AdjO'].values[:len(teams)],
                    'AdjDE': current_season['AdjD'].values[:len(teams)]
                })
                
                self._cache["champion_profile_predictions"] = df
                return df
            except Exception as e2:
                print(f"ERROR creating default predictions: {str(e2)}")
                # Return an empty DataFrame if all else fails
                df = pd.DataFrame(columns=[
                    'SimilarityRank', 'TeamName', 'SimilarityPct', 
                    'ChampionPct', 'FinalFourPct', 'AdjEM', 'AdjOE', 'AdjDE'
                ])
                self._cache["champion_profile_predictions"] = df
                return df
    
    def get_exit_round_predictions(self):
        """
        Load the exit round predictions
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing exit round predictions
        """
        if "exit_round_predictions" in self._cache:
            return self._cache["exit_round_predictions"]
        
        file_path = os.path.join(self.exit_round_dir, "exit_round_predictions.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Exit round predictions not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        self._cache["exit_round_predictions"] = df
        return df
    
    def get_tournament_teams(self):
        """
        Load the predicted tournament teams
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing predicted tournament teams
        """
        if "tournament_teams" in self._cache:
            return self._cache["tournament_teams"]
        
        file_path = os.path.join(self.exit_round_dir, "tournament_teams_predictions.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tournament teams predictions not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        self._cache["tournament_teams"] = df
        return df
    
    def get_combined_predictions(self):
        """
        Combine champion profile and exit round predictions
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing combined predictions
        """
        if "combined_predictions" in self._cache:
            return self._cache["combined_predictions"]
        
        try:
            champion_df = self.get_champion_profile_predictions()
            exit_df = self.get_exit_round_predictions()
            
            # Check if the required columns exist in champion_df
            if 'ChampionPct' not in champion_df.columns:
                champion_df['ChampionPct'] = 0.0
            
            if 'FinalFourPct' not in champion_df.columns:
                champion_df['FinalFourPct'] = 0.0
            
            # Rename columns for consistency before merging
            champion_df = champion_df.rename(columns={
                'ChampionPct': 'ChampionshipPct_ChampProfile',
                'FinalFourPct': 'FinalFourPct_ChampProfile'
            })
            
            # Check if the required columns exist in exit_df
            if 'ChampionshipPct' not in exit_df.columns:
                exit_df['ChampionshipPct'] = 0.0
            
            if 'FinalFourPct' not in exit_df.columns:
                exit_df['FinalFourPct'] = 0.0
            
            # Rename columns for consistency before merging
            exit_df = exit_df.rename(columns={
                'ChampionshipPct': 'ChampionshipPct_ExitRound',
                'FinalFourPct': 'FinalFourPct_ExitRound'
            })
            
            # Merge the dataframes on TeamName
            combined = pd.merge(
                champion_df,
                exit_df[['TeamName', 'Seed', 'PredictedExitRound', 'PredictedExit', 
                         'ChampionshipPct_ExitRound', 'FinalFourPct_ExitRound']],
                on='TeamName',
                how='outer',
                suffixes=('_ChampProfile', '_ExitRound')
            )
            
            # Create combined metrics
            # Average the championship probabilities from both models
            combined['ChampionshipPct_Combined'] = combined[['ChampionshipPct_ChampProfile', 'ChampionshipPct_ExitRound']].mean(axis=1, skipna=True)
            
            # Average the Final Four probabilities from both models
            combined['FinalFourPct_Combined'] = combined[['FinalFourPct_ChampProfile', 'FinalFourPct_ExitRound']].mean(axis=1, skipna=True)
            
            # Create a combined score (weighted average of both models)
            combined['CombinedScore'] = (
                0.5 * combined['SimilarityPct'] + 
                0.5 * (combined['PredictedExitRound'] / 7 * 100)  # Normalize to 0-100 scale
            )
            
            # Sort by combined championship probability
            combined = combined.sort_values('ChampionshipPct_Combined', ascending=False)
            
            self._cache["combined_predictions"] = combined
            return combined
            
        except Exception as e:
            # If there's an error, print it and return None
            print(f"Warning: Error combining predictions: {e}")
            return None
    
    def get_seed_performance(self):
        """
        Load the seed performance data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing seed performance data
        """
        if "seed_performance" in self._cache:
            return self._cache["seed_performance"]
        
        file_path = os.path.join(self.exit_round_dir, "seed_performance.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Seed performance data not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        self._cache["seed_performance"] = df
        return df
    
    def get_historical_champions(self):
        """
        Get a list of historical champions
        
        Returns:
        --------
        list
            List of dictionaries containing champion data
        """
        if "historical_champions" in self._cache:
            return self._cache["historical_champions"]
        
        # Try to load the historical champions data
        file_path = os.path.join(self.champion_profile_dir, "historical_champions.json")
        
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r') as f:
            champions = json.load(f)
        
        self._cache["historical_champions"] = champions
        return champions
        
    def get_tournament_level_analysis(self, round_level):
        """
        Load tournament level analysis data for a specific round
        
        Parameters:
        -----------
        round_level : int or str
            Either the round number (1-7) or the round name
            
        Returns:
        --------
        dict
            Dictionary containing analysis data for the specified round
        """
        # Map of round names
        round_mapping = {
            7: "national_champions",
            6: "championship_game",
            5: "final_four",
            4: "elite_eight",
            3: "sweet_sixteen",
            2: "round_of_32",
            1: "tournament_qualifiers",
            "National Champions": "national_champions",
            "Championship Game": "championship_game",
            "Final Four": "final_four",
            "Elite Eight": "elite_eight",
            "Sweet Sixteen": "sweet_sixteen",
            "Round of 32": "round_of_32",
            "Tournament Qualifiers": "tournament_qualifiers"
        }
        
        # Convert round level to filename
        if isinstance(round_level, int):
            filename = round_mapping.get(round_level)
        else:
            filename = round_mapping.get(round_level.replace(" ", "_").lower())
        
        if not filename:
            raise ValueError(f"Invalid round level: {round_level}")
        
        # Check cache
        cache_key = f"tournament_level_{filename}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try to load the data
        file_path = os.path.join(self.champion_profile_dir, f"{filename}_analysis.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self._cache[cache_key] = data
        return data
    
    def get_all_tournament_level_analysis(self):
        """
        Load all tournament level analysis data
        
        Returns:
        --------
        dict
            Dictionary containing analysis data for all rounds
        """
        if "all_tournament_levels" in self._cache:
            return self._cache["all_tournament_levels"]
        
        # Define round levels
        round_levels = range(1, 8)  # 1-7
        
        # Load data for each round
        all_data = {}
        for round_num in round_levels:
            data = self.get_tournament_level_analysis(round_num)
            if data:
                all_data[round_num] = data 